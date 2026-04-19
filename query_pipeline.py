"""
query_pipeline.py
------------------
Wires the entire retrieval system together.

Full flow:
    User query
        ↓
    QueryClassifier   → RELATIONSHIP | FACT | COMPLEX
        ↓
    GraphRetriever    (if RELATIONSHIP or COMPLEX)
      ├─ Cypher traversal from seed entities
      ├─ SufficiencyChecker
      └─ KGE / FAISS escalation (if available)
    TreeRetriever     (if FACT or COMPLEX)
        ↓
    FusionEngine      → unified scored context + GAR
        ↓
    ReasoningAgent    → sufficiency check → multi-hop → final answer
        ↓
    FinalAnswer with citations

Key classifier behavior:
    FACT     → single atomic value (amount, date, identifier)
    COMPLEX  → anything involving a legal event, order, judgment,
               sequence, justification, or party roles
    RELATIONSHIP → purely about entity connections

Note: the classifier prompt is designed so that most legal queries
classify as COMPLEX, which fires both graph and tree paths.
This is correct — legal questions almost always need both
relationship context AND documentary evidence.
"""

from __future__ import annotations

from pathlib import Path
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

from retrieval.query_classifier import QueryClassifier, QueryType
from retrieval.graph_retriever import GraphRetriever
from retrieval.tree_retriever import TreeRetriever
from retrieval.fusion import FusionEngine
from retrieval.agent import ReasoningAgent, FinalAnswer


class QueryPipeline:
    """
    End-to-end query pipeline.

    Usage:
        pipeline = QueryPipeline()
        pipeline.load(extraction_json_path="cases/celir_llp_vs_midc/extraction.json")
        answer = pipeline.query("Is MIDC justified in demanding ULC charges from CELIR?")
        print(answer.answer)
    """

    def __init__(self):
        self.classifier     = QueryClassifier()
        self.graph_retriever = GraphRetriever()
        self.tree_retriever  = TreeRetriever()
        self.fusion          = FusionEngine()
        self.agent           = ReasoningAgent()
        self._loaded         = False
        self._case_id        = None

    def load(self, extraction_json_path: str = None):
        """
        Load all indexes and models.
        Call once before querying.

        Args:
            extraction_json_path: Path to cases/{case_id}/extraction.json
                                  Used to derive case_id for tree index.
        """
        logger.info("Loading query pipeline...")

        # Derive case_id from extraction path
        if extraction_json_path:
            self._case_id = Path(extraction_json_path).parent.name
        else:
            self._case_id = "celir_case"
            logger.warning(
                "No extraction_json_path provided — "
                f"defaulting case_id to '{self._case_id}'"
            )

        # Load graph retriever (Cypher-first, FAISS optional)
        logger.info("  Loading graph retriever...")
        self.graph_retriever.load(case_id=self._case_id)

        # Load tree retriever (BM25)
        logger.info("  Loading tree retriever (BM25)...")
        try:
            self.tree_retriever.load_index(case_id=self._case_id)
        except FileNotFoundError:
            logger.info(
                f"  No saved BM25 index found for '{self._case_id}' — "
                "building from pages.json..."
            )
            self.tree_retriever.build_index(case_id=self._case_id)

        self._loaded = True
        logger.info(
            f"Pipeline loaded and ready. "
            f"Case: {self._case_id}"
        )

    def query(self, question: str) -> FinalAnswer:
        """
        Answer a question using the full hybrid retrieval pipeline.

        Args:
            question: Natural language question about the case

        Returns:
            FinalAnswer with answer text + citations
        """
        if not self._loaded:
            raise RuntimeError("Call pipeline.load() before querying.")

        logger.info(f"\nQuery: {question}")

        # ── Step 1: Classify ──────────────────────────────────────────────────
        classification = self.classifier.classify(question)
        logger.info(
            f"Classification: {classification.query_type} "
            f"(confidence: {classification.confidence:.2f})"
        )
        logger.info(f"  Reasoning: {classification.reasoning}")
        logger.info(f"  Key entities: {classification.key_entities}")

        # ── Step 2: Retrieve ──────────────────────────────────────────────────
        graph_result = None
        tree_results = []

        if classification.query_type in (QueryType.RELATIONSHIP, QueryType.COMPLEX):
            logger.info("  Running graph path (Cypher-first)...")
            graph_result = self.graph_retriever.search(question, top_k=20)
            path_used = "Cypher+KGE" if graph_result.escalated_to_kge else "Cypher"
            logger.info(
                f"  Graph ({path_used}): {len(graph_result.nodes)} nodes, "
                f"{len(graph_result.subgraph_triples)} triples"
            )

        if classification.query_type in (QueryType.FACT, QueryType.COMPLEX):
            logger.info("  Running tree path (BM25)...")
            tree_results = self.tree_retriever.search(question, top_k=10)
            logger.info(f"  Tree: {len(tree_results)} passages")

        # ── Step 3: Fuse ──────────────────────────────────────────────────────
        logger.info("  Fusing results...")
        fused = self.fusion.fuse(
            question, graph_result, tree_results, self.graph_retriever
        )
        logger.info(
            f"  Fused: {fused.high_confidence_count} HIGH, "
            f"{fused.medium_confidence_count} MEDIUM confidence"
        )

        # ── Step 4: Agent ─────────────────────────────────────────────────────
        logger.info("  Reasoning agent...")
        answer = self.agent.answer(
            question,
            fused,
            graph_retriever=self.graph_retriever,
            tree_retriever=self.tree_retriever,
            fusion_engine=self.fusion,
        )

        logger.info(
            f"  Answer generated: {answer.answer_type}, "
            f"confidence={answer.confidence:.2f}, "
            f"hops={answer.hops_taken}"
        )

        return answer

    def print_answer(self, answer: FinalAnswer):
        """Pretty-print the answer with citations."""
        print(f"\n{'='*70}")
        print(f"QUERY: {answer.query}")
        print(f"{'='*70}")
        print(f"\n{answer.answer}")
        print(f"\n{'─'*70}")
        print(
            f"Type: {answer.answer_type} | "
            f"Confidence: {answer.confidence:.0%} | "
            f"Hops: {answer.hops_taken}"
        )
        if answer.citations:
            print("\nCitations:")
            for c in answer.citations:
                print(
                    f"  [{c.get('source', '?')} p.{c.get('page', '?')}] "
                    f"{c.get('text', '')}"
                )
        print(f"{'='*70}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    pipeline = QueryPipeline()

    # Auto-detect case from cases/ folder
    # Override by passing extraction_json as first arg
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        extraction_json = sys.argv[1]
        question_args = sys.argv[2:]
    else:
        # Try to find extraction.json in most recently modified case
        cases_dir = Path("cases")
        if cases_dir.exists():
            case_dirs = [d for d in cases_dir.iterdir() if d.is_dir()]
            if case_dirs:
                # Most recently modified case
                latest = max(case_dirs, key=lambda d: d.stat().st_mtime)
                extraction_json = str(latest / "extraction.json")
                logger.info(f"Auto-detected case: {latest.name}")
            else:
                extraction_json = "cases/celir_case/extraction.json"
        else:
            extraction_json = "cases/celir_case/extraction.json"
        question_args = sys.argv[1:]

    pipeline.load(extraction_json_path=extraction_json)

    test_questions = [
        "Is MIDC legally justified in demanding ULC charges from CELIR?",
        "What is the relationship between Bafna Motors and Union Bank of India?",
        "What did the Supreme Court order on 21 September 2023?",
        "Who are all the parties involved in this case and what are their roles?",
        "What is the total amount demanded by MIDC and what does it consist of?",
    ]

    if question_args:
        question = " ".join(question_args)
        answer = pipeline.query(question)
        pipeline.print_answer(answer)
    else:
        for question in test_questions:
            answer = pipeline.query(question)
            pipeline.print_answer(answer)