"""
pipeline/tree_builder.py
-------------------------
Build a hierarchical document tree from Markdown produced by pdf_to_markdown.py.

The tree mirrors the legal document structure:
    Document (root)
     ├─ # Title section
     │   ├─ ## Major section (FACTS, GROUNDS...)
     │   │   ├─ ### Subsection
     │   │   │   └─ #### Item
     │   │   └─ ### Subsection
     │   └─ ## Major section
     └─ # Title section

Each TreeNode contains:
    node_id   — 4-digit counter e.g. "0042"
    title     — heading text (stripped of # markers)
    content   — full text under this heading until the next heading
    summary   — filled later by summarize_tree()
    page_num  — page number from <!-- page:N --> markers
    nodes     — list of child TreeNode objects

Output:
    cases/{case_id}/tree.json  — full tree with content
    cases/{case_id}/tree_index.json — lightweight tree (summaries only, no content)

Usage:
    from pipeline.tree_builder import build_tree_from_markdown, save_tree
    tree = build_tree_from_markdown(md_text)
    save_tree(tree, "cases/celir_case/tree.json")
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$")
PAGE_MARKER_RE = re.compile(r"^<!-- page:(\d+) -->$")


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class TreeNode:
    node_id: str
    title: str
    content: str
    summary: str = ""
    page_num: int = 0
    level: int = 0          # 0=root, 1=#, 2=##, 3=###, 4=####
    nodes: list["TreeNode"] = field(default_factory=list)

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.nodes)

    def to_dict(self, include_content: bool = True) -> dict:
        d = {
            "node_id": self.node_id,
            "title": self.title,
            "summary": self.summary,
            "page_num": self.page_num,
            "level": self.level,
        }
        if include_content:
            d["content"] = self.content
        if self.nodes:
            d["nodes"] = [n.to_dict(include_content) for n in self.nodes]
        return d


# ── Builder ────────────────────────────────────────────────────────────────────

def build_tree_from_markdown(markdown: str) -> TreeNode:
    """
    Parse Markdown into a hierarchical TreeNode tree.

    Heading levels:
        #    → level 1
        ##   → level 2
        ###  → level 3
        #### → level 4
    Plain text → content of the current node.
    <!-- page:N --> markers → update current page number.
    """
    root = TreeNode(
        node_id="0000",
        title="Document",
        content="",
        level=0,
    )

    counter = 1
    current_page = 1

    # Stack of (level, node) pairs
    # Stack always starts with root at level 0
    stack: list[tuple[int, TreeNode]] = [(0, root)]

    # Buffer for accumulating content lines
    content_lines: list[str] = []

    def flush_content(node: TreeNode):
        """Assign accumulated content lines to current node."""
        node.content = "\n".join(content_lines).strip()
        content_lines.clear()

    lines = markdown.split("\n")

    for line in lines:
        # Page marker
        page_match = PAGE_MARKER_RE.match(line.strip())
        if page_match:
            current_page = int(page_match.group(1))
            continue

        # Heading line
        header_match = HEADER_RE.match(line)
        if header_match:
            hashes = header_match.group(1)
            title = header_match.group(2).strip()
            level = len(hashes)

            # Flush content to current node
            flush_content(stack[-1][1])

            # Pop stack to correct parent level
            while len(stack) > 1 and stack[-1][0] >= level:
                stack.pop()

            node = TreeNode(
                node_id=f"{counter:04d}",
                title=title,
                content="",
                level=level,
                page_num=current_page,
            )
            counter += 1

            # Attach to parent
            stack[-1][1].nodes.append(node)
            stack.append((level, node))
            continue

        # Plain text — accumulate as content
        content_lines.append(line)

    # Flush remaining content
    flush_content(stack[-1][1])

    logger.info(
        f"Tree built: {root.node_count()} nodes, "
        f"depth {_max_depth(root)}"
    )
    return root


def _max_depth(node: TreeNode, depth: int = 0) -> int:
    if not node.nodes:
        return depth
    return max(_max_depth(c, depth + 1) for c in node.nodes)


# ── Summarization ──────────────────────────────────────────────────────────────

def summarize_tree(node: TreeNode, client, model: str = "gpt-4o-mini"):
    """
    Bottom-up summarization — children first, then parent.
    Each node gets a 2-3 sentence summary using child summaries + own content.

    Args:
        node:   TreeNode to summarize (recursive)
        client: OpenAI client instance
        model:  Model to use for summarization
    """
    # Recurse children first
    for child in node.nodes:
        summarize_tree(child, client, model)

    # Skip root and nodes with no content
    has_content = len(node.content.strip()) > 20
    has_child_summaries = any(c.summary for c in node.nodes)

    if not has_content and not has_child_summaries:
        return

    if has_child_summaries:
        child_text = "\n".join(
            f"- {c.title}: {c.summary}"
            for c in node.nodes if c.summary
        )
        text = (
            f"{node.content}\n\nSubsections:\n{child_text}"
            if has_content else child_text
        )
    else:
        text = node.content

    prompt = f"""Summarize this legal document section in 2-3 sentences.
Be specific — mention parties, amounts, dates, legal provisions where present.

Section: {node.title}

{text[:5000]}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        node.summary = response.choices[0].message.content.strip()
        logger.debug(f"Summarized: {node.title[:50]}")
    except Exception as e:
        logger.warning(f"Summarization failed for {node.title}: {e}")


# ── Save / Load ────────────────────────────────────────────────────────────────

def save_tree(tree: TreeNode, path: str | Path, include_content: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree.to_dict(include_content), f, indent=2, ensure_ascii=False)
    logger.info(f"Tree saved → {path}")


def load_tree(path: str | Path) -> TreeNode:
    with open(path, "r", encoding="utf-8") as f:
        return _dict_to_node(json.load(f))


def _dict_to_node(d: dict) -> TreeNode:
    return TreeNode(
        node_id=d["node_id"],
        title=d["title"],
        content=d.get("content", ""),
        summary=d.get("summary", ""),
        page_num=d.get("page_num", 0),
        level=d.get("level", 0),
        nodes=[_dict_to_node(c) for c in d.get("nodes", [])],
    )


def create_node_map(root: TreeNode) -> dict[str, TreeNode]:
    node_map: dict[str, TreeNode] = {}
    def dfs(node: TreeNode):
        node_map[node.node_id] = node
        for child in node.nodes:
            dfs(child)
    dfs(root)
    return node_map


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python tree_builder.py <markdown_file>")
        sys.exit(1)

    md_path = Path(sys.argv[1])
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    tree = build_tree_from_markdown(md_text)
    out = md_path.with_suffix(".tree.json")
    save_tree(tree, out)
    print(f"Tree saved to {out}")
    print(f"Total nodes: {tree.node_count()}")

    # Show first 5 nodes
    def show(node, depth=0):
        if depth > 3:
            return
        print("  " * depth + f"[{node.node_id}] {node.title[:60]}")
        for c in node.nodes[:3]:
            show(c, depth + 1)
        if len(node.nodes) > 3:
            print("  " * (depth + 1) + f"... {len(node.nodes) - 3} more")

    show(tree)