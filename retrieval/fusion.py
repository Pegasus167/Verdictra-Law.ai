"""
retrieval/fusion.py
--------------------
Fusion layer — merges results from graph path and tree path.

Fusion logic:
    1. Collect all graph nodes + tree passages
    2. Find overlaps (same entity appears in both)
    3. Assign confidence levels:
       - HIGH:   appears in both graph + tree
       - MEDIUM: appears in one path only
       - LOW:    appears but with low individual scores
    4. Return unified context ready for LLM agent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from retrieval.graph_retriever import GraphNode, GraphSearchResult
from retrieval.tree_retriever import TreeSearchResult

logger = logging.getLogger(__name__)

# ── Relationship strength ──────────────────────────────────────────────────────
# CO_OCCURS_WITH is a weak co-occurrence signal — two entities appeared on the
# same page. It does NOT imply any semantic relationship between them.
# All other relationship types are typed (extracted or upgraded) and carry
# actual semantic meaning.
#
# This set is used universally — no case-specific names anywhere.
# Any relationship type not in WEAK_RELATIONS is treated as typed/strong.
WEAK_RELATIONS = {"CO_OCCURS_WITH", "CO_OCCURS", "COOCCURS_WITH"}


def _relation_strength(rel: str) -> str:
    """Return 'weak' or 'typed' for a relationship type string."""
    return "weak" if rel.upper() in WEAK_RELATIONS else "typed"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"       # Confirmed by both graph + tree
    MEDIUM = "MEDIUM"   # Confirmed by one path
    LOW = "LOW"         # Weak signal


@dataclass
class FusedResult:
    """A single fused result with evidence from both paths."""
    entity_name: str
    confidence_level: ConfidenceLevel

    # Graph evidence
    graph_node: GraphNode | None = None
    graph_score: float = 0.0

    # Tree evidence
    tree_passages: list[TreeSearchResult] = field(default_factory=list)
    tree_score: float = 0.0

    # Combined
    final_score: float = 0.0
    citations: list[str] = field(default_factory=list)


@dataclass
class FusedContext:
    """Complete fused context passed to the LLM agent."""
    query: str
    results: list[FusedResult]
    subgraph_triples: list[tuple[str, str, str]]
    high_confidence_count: int
    medium_confidence_count: int

    # Derived counts — set by FusionEngine, used by agent sufficiency check
    typed_triple_count: int = 0     # Triples with semantic relationship types
    weak_triple_count: int = 0      # CO_OCCURS_WITH triples (positional only)

    gar_triples: list[tuple[str, str, str, str, int]] = field(default_factory=list)
    # (head, relation, tail, source_passage_citation, page_number)
    # These are graph relationships grounded in specific document passages

    def to_prompt_context(self) -> str:
        """
        Convert fused results to a structured string for the LLM prompt.

        Key design decision: relationship strength is annotated explicitly.
        - [typed] relationships carry semantic meaning → use as evidence
        - [weak / co-occurrence only] means two entities appeared on the
          same page — this is NOT evidence of any relationship between them.
          Do not use weak relationships to infer causation or justification.
        """
        lines = [
            f"Query: {self.query}",
            "",
            f"Retrieved {len(self.results)} relevant items "
            f"({self.high_confidence_count} HIGH confidence, "
            f"{self.medium_confidence_count} MEDIUM confidence)",
            f"Graph triples: {self.typed_triple_count} typed "
            f"(semantic), {self.weak_triple_count} weak (co-occurrence only)",
            "",
            "=== KNOWLEDGE GRAPH EVIDENCE ===",
            "Relationship strength legend:",
            "  [typed]  — has semantic meaning, use as evidence",
            "  [weak]   — co-occurrence only (same page), NOT proof of relationship",
            "",
        ]

        # Graph triples — annotated by strength
        if self.subgraph_triples:
            lines.append("Relationships:")
            for head, rel, tail in self.subgraph_triples[:30]:
                strength = _relation_strength(rel)
                lines.append(f"  [{strength}] ({head}) --[{rel}]--> ({tail})")
        else:
            lines.append("No direct relationships found between retrieved entities.")

        lines.append("")
        lines.append("=== DOCUMENT EVIDENCE ===")

        for result in self.results:
            conf_marker = {
                ConfidenceLevel.HIGH: "[HIGH]",
                ConfidenceLevel.MEDIUM: "[MEDIUM]",
                ConfidenceLevel.LOW: "[LOW]",
            }[result.confidence_level]

            lines.append(
                f"\n{conf_marker} {result.entity_name}"
            )

            if result.graph_node:
                lines.append(
                    f"  Graph: {result.graph_node.schema_type} "
                    f"(score: {result.graph_score:.2f})"
                )

            for passage in result.tree_passages[:3]:
                lines.append(f"  Document [{passage.citation}]:")
                if passage.passage.relevant_lines:
                    for line in passage.passage.relevant_lines[:5]:
                        lines.append(f"    \"{line[:200]}\"")
                # Also include a window of the full page text
                if passage.passage.text:
                    lines.append(f"  Full page text (first 800 chars):")
                    lines.append(f"    {passage.passage.text[:800]}")
                    
        if self.gar_triples:
            lines.append("")
            lines.append("=== GRAPH-AUGMENTED EVIDENCE (document-grounded) ===")
            lines.append("These relationships were found by looking up entities")
            lines.append("mentioned in the retrieved document passages.")
            lines.append("They are grounded in specific pages — use as strong evidence.")
            lines.append("")
            for head, rel, tail, citation, page in self.gar_triples:
                strength = _relation_strength(rel)
                lines.append(
                    f"  [{strength}] ({head}) --[{rel}]--> ({tail})"
                    f"  ← grounded in {citation}"
                )

        return "\n".join(lines)


# ── Fusion Engine ──────────────────────────────────────────────────────────────

class FusionEngine:
    """
    Merges graph and tree retrieval results.

    Usage:
        engine = FusionEngine()
        fused = engine.fuse(query, graph_result, tree_results)
        context_str = fused.to_prompt_context()
    """

    def __init__(
        self,
        graph_weight: float = None,
        tree_weight: float = None,
    ):
        from config import settings
        self.graph_weight = graph_weight or settings.fusion_graph_weight
        self.tree_weight = tree_weight or settings.fusion_tree_weight

    def fuse(
        self,
        query: str,
        graph_result: GraphSearchResult | None,
        tree_results: list[TreeSearchResult],
        graph_retriever=None,  
    ) -> FusedContext:
        """
        Fuse graph and tree results into unified context.
        """
        fused_map: dict[str, FusedResult] = {}

        # Process graph results
        if graph_result is not None:
            for node in graph_result.nodes:
                key = node.canonical_name
                if key not in fused_map:
                    fused_map[key] = FusedResult(
                        entity_name=key,
                        confidence_level=ConfidenceLevel.MEDIUM,
                    )
                fused_map[key].graph_node = node
                fused_map[key].graph_score = node.node_score

        # Process tree results
        for tree_result in tree_results:
            passage_text = tree_result.passage.text.lower()

            matched_entities = []
            if graph_result is not None:
                for node in graph_result.nodes:
                    name_parts = node.canonical_name.replace("_", " ").split()
                    if any(part in passage_text for part in name_parts if len(part) > 3):
                        matched_entities.append(node.canonical_name)

            if matched_entities:
                for entity_name in matched_entities:
                    if entity_name not in fused_map:
                        fused_map[entity_name] = FusedResult(
                            entity_name=entity_name,
                            confidence_level=ConfidenceLevel.MEDIUM,
                        )
                    fused_map[entity_name].tree_passages.append(tree_result)
                    fused_map[entity_name].tree_score = max(
                        fused_map[entity_name].tree_score,
                        tree_result.score,
                    )
            else:
                key = f"passage_p{tree_result.passage.page_number}"
                fused_map[key] = FusedResult(
                    entity_name=key,
                    confidence_level=ConfidenceLevel.MEDIUM,
                    tree_passages=[tree_result],
                    tree_score=tree_result.score,
                )

        # Assign confidence levels + final scores
        results = []
        for key, result in fused_map.items():
            has_graph = result.graph_node is not None
            has_tree = len(result.tree_passages) > 0

            if has_graph and has_tree:
                result.confidence_level = ConfidenceLevel.HIGH
            elif has_graph or has_tree:
                result.confidence_level = ConfidenceLevel.MEDIUM
            else:
                result.confidence_level = ConfidenceLevel.LOW

            result.final_score = (
                self.graph_weight * result.graph_score
                + self.tree_weight * result.tree_score
            )

            if result.graph_node:
                result.citations.append(
                    f"Graph: {result.graph_node.schema_type} "
                    f"[p.{result.graph_node.source_page}]"
                )
            for p in result.tree_passages:
                result.citations.append(p.citation)

            results.append(result)

        # Sort by confidence then score
        confidence_order = {
            ConfidenceLevel.HIGH: 0,
            ConfidenceLevel.MEDIUM: 1,
            ConfidenceLevel.LOW: 2,
        }
        results.sort(
            key=lambda r: (confidence_order[r.confidence_level], -r.final_score)
        )

        subgraph = graph_result.subgraph_triples if graph_result else []

        # Count typed vs weak triples for agent sufficiency awareness
        typed_count = sum(
            1 for _, rel, _ in subgraph
            if _relation_strength(rel) == "typed"
        )
        weak_count = len(subgraph) - typed_count

        fused_ctx = FusedContext(
            query=query,
            results=results,
            subgraph_triples=subgraph,
            high_confidence_count=sum(
                1 for r in results if r.confidence_level == ConfidenceLevel.HIGH
            ),
            medium_confidence_count=sum(
                1 for r in results if r.confidence_level == ConfidenceLevel.MEDIUM
            ),
            typed_triple_count=typed_count,
            weak_triple_count=weak_count,
            gar_triples=[],
        )

        # GAR — extract entity names from tree passages, look up in graph
        if graph_retriever is not None and tree_results:
            fused_ctx.gar_triples = self._run_gar(
                tree_results, graph_retriever
            )
            logger.info(
                f"GAR: {len(fused_ctx.gar_triples)} document-grounded triples added"
            )

        return fused_ctx
    
    def _run_gar(
        self,
        tree_results: list[TreeSearchResult],
        graph_retriever,
    ) -> list[tuple[str, str, str, str, int]]:
        """
        Extract entity names from tree passages and look up their
        relationships in Neo4j. Returns grounded triples with citations.
        """
        # Extract candidate entity names from passage text
        # Match against graph's canonical names using token overlap
        all_canonical = graph_retriever._all_canonical_names
        gar_triples = []
        seen_triples = set()

        for tree_result in tree_results[:5]:  # top 5 passages only
            passage_text = tree_result.passage.text.lower()
            citation = tree_result.citation
            page = tree_result.passage.page_number

            # Find canonical names mentioned in this passage
            mentioned = []
            for name in all_canonical:
                name_clean = name.replace("_", " ").lower()
                # Check if meaningful part of name appears in passage
                parts = [p for p in name_clean.split() if len(p) > 3]
                if parts and any(p in passage_text for p in parts):
                    mentioned.append(name)

            if not mentioned:
                continue

            # Look up typed relationships for these entities
            triples = graph_retriever.gar_lookup(mentioned[:10])
            for head, rel, tail in triples:
                key = (head, rel, tail)
                if key not in seen_triples:
                    seen_triples.add(key)
                    gar_triples.append((head, rel, tail, citation, page))

        return gar_triples[:30]  # cap at 30 grounded triples