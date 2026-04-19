"""
retrieval/tree_retriever.py
----------------------------
Document Tree path of the hybrid retrieval system.

Responsibilities:
    - Load FULL PAGE TEXT from cases/{case_id}/pages.json
    - Build BM25 index over full page text (not entity snippets)
    - At query time: return top-K passages with exact page citations

Why full page text matters:
    The old version used 150-char entity context snippets which meant
    BM25 could find the right page but the passage returned to the agent
    was too short to contain the actual answer.

    e.g. "What is the total amount?" → page 7 found correctly, but
    the 150-char snippet missed the full breakdown. Now the entire
    page text is indexed and returned.

Why BM25 for the tree path:
    Legal documents are dense with specific terms: case numbers,
    section references, amounts, dates. BM25 rewards exact matches
    on these terms — "Article 226" or "₹14,27,26,992" will score
    correctly even if semantically similar text doesn't exist.

Usage:
    retriever = TreeRetriever()
    retriever.build_index(case_id="celir_case")
    results = retriever.search("ULC transfer charges exemption", top_k=5)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank_bm25 import BM25Okapi
from config import settings

logger = logging.getLogger(__name__)


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class Passage:
    """A retrievable passage from the document tree."""
    source_pdf: str
    page_number: int
    section_title: str
    text: str
    word_count: int

    # Set at retrieval time
    bm25_score: float = 0.0
    relevant_lines: list[str] = None


@dataclass
class TreeSearchResult:
    """Result from tree retrieval."""
    passage: Passage
    score: float
    matched_terms: list[str]
    citation: str


# ── Tree Retriever ─────────────────────────────────────────────────────────────

class TreeRetriever:
    """
    BM25-based document tree retriever using full page text.

    Usage:
        retriever = TreeRetriever()

        # Build from full page text (fast — no LLM needed)
        retriever.build_index(case_id="celir_case")

        # Or load existing index
        retriever.load_index(case_id="celir_case")

        # Search
        results = retriever.search("MIDC demand notice ULC charges", top_k=10)
        for r in results:
            print(r.citation, r.score)
            print(r.passage.text[:500])
    """

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._passages: list[Passage] = []
        self._tokenized_corpus: list[list[str]] = []
        self._case_id: str | None = None

    # ── Index paths ────────────────────────────────────────────────────────────

    def _index_path(self, case_id: str) -> Path:
        return settings.case_tree_index(case_id) / "bm25_index.pkl"

    def _passages_path(self, case_id: str) -> Path:
        return settings.case_tree_index(case_id) / "passages.json"

    def _pages_path(self, case_id: str) -> Path:
        """Full OCR page text saved by ingestion.py."""
        return settings.case_path(case_id) / "pages.json"

    # ── Index Building ─────────────────────────────────────────────────────────

    def build_index(self, case_id: str, save: bool = True):
        """
        Build BM25 index from full page text in pages.json.

        Falls back to extraction.json entity contexts if pages.json
        doesn't exist (e.g. for the existing CELIR case before re-ingestion).
        """
        self._case_id = case_id
        pages_path = self._pages_path(case_id)

        if pages_path.exists():
            logger.info(f"Loading full page text from {pages_path}...")
            passages = self._load_from_pages_json(pages_path, case_id)
        else:
            logger.warning(
                f"pages.json not found at {pages_path} — "
                "falling back to entity context snippets from extraction.json. "
                "Re-run ingestion.py to get full page text."
            )
            extraction_path = settings.case_extraction(case_id)
            if not extraction_path.exists():
                raise FileNotFoundError(
                    f"Neither pages.json nor extraction.json found for {case_id}"
                )
            passages = self._load_from_extraction(extraction_path)

        if not passages:
            raise ValueError(f"No passages found for case {case_id}.")

        self._passages = passages
        self._tokenized_corpus = [self._tokenize(p.text) for p in passages]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(f"BM25 index built: {len(passages)} pages indexed")

        if save:
            self._save_index(case_id)

    def _load_from_pages_json(
        self, pages_path: Path, case_id: str
    ) -> list[Passage]:
        """Load full page text from pages.json (preferred)."""
        with open(pages_path, "r", encoding="utf-8") as f:
            pages_data = json.load(f)

        passages = []
        for p in pages_data:
            text = p.get("text", "").strip()
            if not text:
                continue
            passages.append(Passage(
                source_pdf=p.get("source_pdf", ""),
                page_number=p.get("page_number", 0),
                section_title="",
                text=text,
                word_count=p.get("word_count", len(text.split())),
            ))

        logger.info(
            f"Loaded {len(passages)} full pages "
            f"(avg {sum(p.word_count for p in passages)//max(len(passages),1)} words/page)"
        )
        return passages

    def _load_from_extraction(self, extraction_path: Path) -> list[Passage]:
        """
        Fallback: load from entity context snippets in extraction.json.
        Groups all entity contexts by page.
        """
        with open(extraction_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_pdf = data.get("source_pdf", "unknown.pdf")
        page_texts: dict[int, list[str]] = {}

        for entity in data.get("entities", []):
            page = entity.get("source_page", 0)
            context = entity.get("context", "").strip()
            if context and page > 0:
                page_texts.setdefault(page, []).append(context)

        passages = []
        for page_num in sorted(page_texts.keys()):
            texts = page_texts[page_num]
            full_text = " ... ".join(dict.fromkeys(texts))
            passages.append(Passage(
                source_pdf=source_pdf,
                page_number=page_num,
                section_title="",
                text=full_text,
                word_count=len(full_text.split()),
            ))

        return passages

    def _save_index(self, case_id: str):
        """Save BM25 index and passages to cases/{case_id}/tree_index/."""
        index_dir = settings.case_tree_index(case_id)
        index_dir.mkdir(parents=True, exist_ok=True)

        with open(self._index_path(case_id), "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "tokenized_corpus": self._tokenized_corpus,
            }, f)

        passages_data = [
            {
                "source_pdf": p.source_pdf,
                "page_number": p.page_number,
                "section_title": p.section_title,
                "text": p.text,
                "word_count": p.word_count,
            }
            for p in self._passages
        ]
        with open(self._passages_path(case_id), "w", encoding="utf-8") as f:
            json.dump(passages_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Tree index saved → {index_dir}")

    def load_index(self, case_id: str = None):
        """
        Load existing BM25 index from disk.
        case_id defaults to 'celir_case' for backward compatibility.
        """
        case_id = case_id or "celir_case"
        self._case_id = case_id

        index_path = self._index_path(case_id)
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index found at {index_path}. "
                "Run build_index() first."
            )

        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._tokenized_corpus = data["tokenized_corpus"]

        with open(self._passages_path(case_id), "r", encoding="utf-8") as f:
            passages_data = json.load(f)

        self._passages = [
            Passage(
                source_pdf=p["source_pdf"],
                page_number=p["page_number"],
                section_title=p.get("section_title", ""),
                text=p["text"],
                word_count=p["word_count"],
            )
            for p in passages_data
        ]

        logger.info(f"Tree index loaded: {len(self._passages)} passages")

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[TreeSearchResult]:
        """
        Search using BM25. Returns top-K passages with full page text.

        Key improvement over old version:
        - Returns full page text (not 150-char snippets)
        - relevant_lines extracts the most answer-dense lines for citations
        """
        if self._bm25 is None:
            raise RuntimeError(
                "Index not loaded. Call build_index() or load_index()."
            )

        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            passage = self._passages[idx]
            passage.bm25_score = float(scores[idx])

            matched = self._find_matched_terms(query_tokens, passage.text)
            relevant_lines = self._extract_relevant_lines(
                passage.text, query_tokens
            )
            passage.relevant_lines = relevant_lines

            citation = f"Page {passage.page_number}"
            if passage.section_title:
                citation += f", Section: {passage.section_title}"
            citation += f", {passage.source_pdf}"

            results.append(TreeSearchResult(
                passage=passage,
                score=float(scores[idx]),
                matched_terms=matched,
                citation=citation,
            ))

        logger.debug(
            f"Tree search '{query[:50]}': {len(results)} results"
        )
        return results

    def search_by_page(self, page_number: int) -> Passage | None:
        for p in self._passages:
            if p.page_number == page_number:
                return p
        return None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize for BM25.
        Preserves legal tokens: amounts (₹14,27,26,992), section refs,
        case numbers, dates.
        """
        if not text:
            return []
        text = text.lower()
        # Preserve amounts and case numbers with dots/commas
        tokens = re.findall(r'\b[\w.,/-]+\b', text)
        tokens = [t.strip(".,") for t in tokens if len(t) > 1]
        return tokens

    def _find_matched_terms(
        self, query_tokens: list[str], text: str
    ) -> list[str]:
        text_lower = text.lower()
        return [t for t in query_tokens if t in text_lower]

    def _extract_relevant_lines(
        self, text: str, query_tokens: list[str], max_lines: int = 5
    ) -> list[str]:
        """
        Extract the most relevant lines from a full page.
        Returns up to 5 lines (increased from 3) for richer context.
        """
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return []

        def line_score(line: str) -> int:
            line_lower = line.lower()
            return sum(1 for t in query_tokens if t in line_lower)

        scored = [(line_score(l), l) for l in lines]
        scored.sort(reverse=True)
        return [l for _, l in scored[:max_lines] if _ > 0]


# ── Convenience functions ──────────────────────────────────────────────────────

def build_tree_index(case_id: str) -> TreeRetriever:
    retriever = TreeRetriever()
    retriever.build_index(case_id)
    return retriever


def load_tree_index(case_id: str = "celir_case") -> TreeRetriever:
    retriever = TreeRetriever()
    retriever.load_index(case_id)
    return retriever


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    case_id = sys.argv[1] if len(sys.argv) > 1 else "celir_case"

    pages_path = settings.case_path(case_id) / "pages.json"
    if pages_path.exists():
        print(f"Building index from full page text ({pages_path})...")
    else:
        print(
            f"WARNING: pages.json not found. "
            "Falling back to entity snippets. "
            "Re-run ingestion.py to get full page text."
        )

    retriever = build_tree_index(case_id)

    test_queries = [
        "total amount demanded MIDC ULC charges",
        "parties involved case CELIR MIDC",
        "Supreme Court order September 2023",
        "Bafna Motors Union Bank mortgage",
        "ULC transfer charges exemption",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        results = retriever.search(query, top_k=3)
        for r in results:
            print(f"  [{r.citation}] score={r.score:.2f}")
            print(f"  Matched: {r.matched_terms}")
            if r.passage.relevant_lines:
                for line in r.passage.relevant_lines[:2]:
                    print(f"  → {line[:120]}")