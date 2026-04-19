"""
retrieval/graph_retriever.py
-----------------------------
Graph path of the hybrid retrieval system.

Strategy: Priority-ordered retrieval — Cypher first, KGE only if needed.

Flow:
    1. Extract candidate entity names from the query via token overlap
       against all Neo4j canonical names for this case.
    2. Cypher traversal — 1–3 hops from seed nodes, filtered by case_id.
    3. SufficiencyChecker evaluates the Cypher result.
    4. KGE path (only if escalated): FAISS → seeds → Cypher → merge.
    5. Score every node: Score = α·Relevance + β·Importance + γ·Confidence
    6. Return top-k scored nodes + subgraph triples.

case_id partitioning:
    Every Cypher query filters WHERE n.case_id = $case_id.
    This ensures queries only traverse the current case's subgraph.
    Multiple cases can coexist in the same Neo4j instance safely.
    Cross-case queries are possible via cross_case_search() in graph_builder.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from neo4j import GraphDatabase

from config import settings

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = Path("data/embeddings/graph_entities.faiss")
ENTITY_MAP_PATH  = Path("data/embeddings/entity_map.pkl")

DEFAULT_HOPS = 2
MAX_SEEDS    = 5

# ── Sufficiency thresholds ─────────────────────────────────────────────────────
MIN_NODES            = 5
MIN_CONFIDENT_NODES  = 2
CONFIDENCE_THRESHOLD = 0.6
MIN_TRIPLES          = 1


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    canonical_name: str
    text: str
    schema_type: str
    source_pdf: str
    source_page: int
    confidence: float
    degree: int = 0
    embedding_similarity: float = 0.0
    node_score: float = 0.0
    relationships: list[dict] = field(default_factory=list)


@dataclass
class GraphSearchResult:
    nodes: list[GraphNode]
    subgraph_triples: list[tuple[str, str, str]]
    query_embedding: list[float]
    escalated_to_kge: bool = False


@dataclass
class SufficiencyReport:
    is_sufficient: bool
    node_count: int
    confident_node_count: int
    triple_count: int
    reason: str


class SufficiencyChecker:
    def check(self, result: GraphSearchResult) -> SufficiencyReport:
        node_count      = len(result.nodes)
        triple_count    = len(result.subgraph_triples)
        confident_count = sum(
            1 for n in result.nodes
            if (n.confidence or 0.0) >= CONFIDENCE_THRESHOLD
        )

        failures = []
        if node_count < MIN_NODES:
            failures.append(f"only {node_count} nodes (need {MIN_NODES})")
        if confident_count < MIN_CONFIDENT_NODES:
            failures.append(
                f"only {confident_count} confident nodes "
                f"(need {MIN_CONFIDENT_NODES})"
            )
        if triple_count < MIN_TRIPLES:
            failures.append(f"only {triple_count} triples (need {MIN_TRIPLES})")

        is_sufficient = len(failures) == 0
        reason = (
            "Cypher result sufficient — skipping KGE"
            if is_sufficient
            else "Cypher insufficient: " + "; ".join(failures)
        )
        return SufficiencyReport(
            is_sufficient=is_sufficient,
            node_count=node_count,
            confident_node_count=confident_count,
            triple_count=triple_count,
            reason=reason,
        )


# ── Graph Retriever ────────────────────────────────────────────────────────────

class GraphRetriever:
    """
    Cypher-traversal graph retriever with optional FAISS/KGE upgrade.
    All queries are scoped to a specific case_id.

    Usage:
        retriever = GraphRetriever()
        retriever.load(case_id="celir_llp_vs_midc")
        results = retriever.search("MIDC demand charges CELIR", top_k=10)
    """

    def __init__(self):
        self._neo4j_driver       = None
        self._case_id: str       = ""
        self._all_canonical_names: list[str] = []
        self._high_degree_nodes: list[dict]  = []
        self._schema_type_map: dict[str, list[str]] = {}
        self._sufficiency_checker = SufficiencyChecker()

        # FAISS — optional
        self._faiss_index  = None
        self._entity_to_idx: dict[str, int] = {}
        self._idx_to_entity: dict[int, str] = {}
        self._text_encoder = None
        self._faiss_loaded = False

    # ── Load ───────────────────────────────────────────────────────────────────

    def load(self, case_id: str = ""):
        """
        Connect to Neo4j and cache entity names for this case.

        Args:
            case_id: Case to load. All queries will be scoped to this case.
                     If empty, falls back to unscoped queries (legacy mode).
        """
        self._case_id = case_id
        driver = self._get_neo4j()

        # Build case filter for Cypher
        case_filter = "AND n.case_id = $case_id" if case_id else ""
        params = {"case_id": case_id} if case_id else {}

        with driver.session(database=settings.neo4j_database) as session:
            # Cache all canonical names for this case
            result = session.run(
                f"MATCH (n) WHERE n.canonicalName IS NOT NULL "
                f"{case_filter} "
                f"RETURN n.canonicalName AS name",
                **params,
            )
            self._all_canonical_names = [
                record["name"] for record in result if record["name"]
            ]

            # Cache top-20 high-degree nodes as fallback seeds
            result2 = session.run(f"""
                MATCH (n)
                WHERE n.canonicalName IS NOT NULL {case_filter}
                OPTIONAL MATCH (n)-[r]-()
                WITH n, count(r) AS degree
                ORDER BY degree DESC
                LIMIT 20
                RETURN n.canonicalName AS name, n.schemaType AS type, degree
            """, **params)
            self._high_degree_nodes = [
                {"name": r["name"], "type": r["type"], "degree": r["degree"]}
                for r in result2 if r["name"]
            ]

            # Cache schema type → canonical names mapping
            result3 = session.run(f"""
                MATCH (n)
                WHERE n.canonicalName IS NOT NULL {case_filter}
                RETURN n.canonicalName AS name, n.schemaType AS type
            """, **params)
            self._schema_type_map = {}
            for r in result3:
                t = (r["type"] or "").lower()
                if t:
                    self._schema_type_map.setdefault(t, []).append(r["name"])

        logger.info(
            f"Graph retriever loaded for case '{case_id}': "
            f"{len(self._all_canonical_names)} canonical names, "
            f"{len(self._high_degree_nodes)} high-degree nodes cached."
        )

        # Try loading FAISS — look in case-specific embeddings folder first
        self._try_load_faiss(case_id)

    def _try_load_faiss(self, case_id: str = ""):
        """Load FAISS index if available. Non-fatal if missing."""
        if not _FAISS_AVAILABLE:
            return

        # Check case-specific embeddings first
        faiss_path    = FAISS_INDEX_PATH
        entity_path   = ENTITY_MAP_PATH

        if case_id:
            case_faiss_config = settings.case_embeddings(case_id) / "faiss_config.json"
            if case_faiss_config.exists():
                import json
                with open(case_faiss_config, "r") as f:
                    cfg = json.load(f)
                faiss_path  = Path(cfg["faiss_path"])
                entity_path = Path(cfg["entity_map_path"])

        if not faiss_path.exists():
            logger.info(f"No FAISS index found for case '{case_id}' — Cypher-only mode.")
            return

        try:
            self._faiss_index = faiss.read_index(str(faiss_path))
            with open(entity_path, "rb") as f:
                maps = pickle.load(f)
            self._entity_to_idx = maps["entity_to_idx"]
            self._idx_to_entity = maps["idx_to_entity"]
            self._text_encoder  = SentenceTransformer(settings.embedding_model)
            self._faiss_loaded  = True
            logger.info(
                f"FAISS index loaded for case '{case_id}': "
                f"{self._faiss_index.ntotal} entities."
            )
        except Exception as e:
            logger.warning(f"FAISS load failed ({e}) — Cypher-only mode.")

    # ── Neo4j ──────────────────────────────────────────────────────────────────

    def _get_neo4j(self):
        if self._neo4j_driver is None:
            self._neo4j_driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
            )
        return self._neo4j_driver

    def _case_params(self, extra: dict = None) -> dict:
        """Build params dict with case_id included."""
        p = {"case_id": self._case_id} if self._case_id else {}
        if extra:
            p.update(extra)
        return p

    def _case_where(self, alias: str = "n") -> str:
        """Return Cypher WHERE clause fragment for case_id filter."""
        return f"AND {alias}.case_id = $case_id" if self._case_id else ""

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = None,
        entity_filter: list[str] = None,
        hops: int = DEFAULT_HOPS,
    ) -> GraphSearchResult:
        """Find relevant graph nodes scoped to the current case_id."""
        top_k = top_k or settings.top_k_graph

        seed_names   = self._match_query_to_entities(query)
        logger.info(f"Cypher seeds: {seed_names}")

        cypher_result = self._run_cypher_path(
            query, seed_names, top_k, entity_filter, hops,
            faiss_similarities={}, query_embedding=[],
        )

        report = self._sufficiency_checker.check(cypher_result)
        logger.info(f"Sufficiency: {report.reason}")

        if report.is_sufficient or not self._faiss_loaded:
            return cypher_result

        # KGE escalation
        logger.info("Escalating to KGE / FAISS...")
        faiss_similarities, query_embedding = self._faiss_search(query, top_k * 2)

        kge_seeds     = list(faiss_similarities.keys())[:MAX_SEEDS]
        merged_seeds  = list(dict.fromkeys(seed_names + kge_seeds))
        logger.info(f"KGE seeds (merged): {merged_seeds}")

        kge_result = self._run_cypher_path(
            query, merged_seeds, top_k, entity_filter, hops,
            faiss_similarities=faiss_similarities,
            query_embedding=query_embedding,
        )
        kge_result.escalated_to_kge = True

        kge_names    = {n.canonical_name for n in kge_result.nodes}
        extra_nodes  = [n for n in cypher_result.nodes if n.canonical_name not in kge_names]
        kge_result.nodes = (kge_result.nodes + extra_nodes)[:top_k]

        all_triples = list(dict.fromkeys(
            kge_result.subgraph_triples + cypher_result.subgraph_triples
        ))
        kge_result.subgraph_triples = all_triples
        return kge_result

    # ── GAR lookup ─────────────────────────────────────────────────────────────

    def gar_lookup(self, entity_names: list[str]) -> list[tuple[str, str, str]]:
        """
        Graph-Augmented Retrieval — fetch typed relationships for entity names,
        scoped to the current case_id.
        """
        if not entity_names:
            return []

        driver = self._get_neo4j()
        with driver.session(database=settings.neo4j_database) as session:
            case_filter = (
                "AND h.case_id = $case_id AND t.case_id = $case_id"
                if self._case_id else ""
            )
            result = session.run(f"""
                MATCH (h)-[r]->(t)
                WHERE (h.canonicalName IN $names OR t.canonicalName IN $names)
                  AND NOT type(r) IN ['CO_OCCURS_WITH', 'CO_OCCURS', 'COOCCURS_WITH']
                  {case_filter}
                RETURN
                    h.canonicalName AS head,
                    type(r)         AS relation,
                    t.canonicalName AS tail
                LIMIT 50
            """, self._case_params({"names": entity_names}))
            return [
                (record["head"], record["relation"], record["tail"])
                for record in result
            ]

    # ── Internal: shared Cypher execution ─────────────────────────────────────

    def _run_cypher_path(
        self,
        query: str,
        seed_names: list[str],
        top_k: int,
        entity_filter: list[str] | None,
        hops: int,
        faiss_similarities: dict[str, float],
        query_embedding: list[float],
    ) -> GraphSearchResult:
        if not seed_names:
            logger.warning("No seed entities found for query — returning empty.")
            return GraphSearchResult(
                nodes=[], subgraph_triples=[],
                query_embedding=query_embedding,
            )

        raw_nodes = self._traverse(seed_names[:MAX_SEEDS], hops, entity_filter)

        if not raw_nodes:
            logger.warning("Traversal returned no nodes.")
            return GraphSearchResult(
                nodes=[], subgraph_triples=[],
                query_embedding=query_embedding,
            )

        query_tokens  = _tokenize(query)
        scored_nodes  = self._score_nodes(raw_nodes, query_tokens, faiss_similarities)
        scored_nodes.sort(key=lambda n: n.node_score, reverse=True)
        scored_nodes  = scored_nodes[:top_k]

        subgraph = self._fetch_subgraph([n.canonical_name for n in scored_nodes])

        return GraphSearchResult(
            nodes=scored_nodes,
            subgraph_triples=subgraph,
            query_embedding=query_embedding,
        )

    # ── Entity Matching ────────────────────────────────────────────────────────

    def _match_query_to_entities(self, query: str) -> list[str]:
        """Three-tier entity matching: token overlap → type keywords → high-degree."""
        query_tokens = _tokenize(query)

        # Tier 1 — token overlap
        scored: list[tuple[str, float]] = []
        for name in self._all_canonical_names:
            name_tokens = _tokenize(name)
            if not name_tokens:
                continue
            overlap = len(query_tokens & name_tokens)
            if overlap > 0:
                score = overlap / len(query_tokens | name_tokens)
                scored.append((name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        seeds = [name for name, _ in scored[:MAX_SEEDS]]
        if seeds:
            return seeds

        # Tier 2 — schema type keywords
        type_keywords = {
            "parties": ["organization", "person", "financialinstitution", "government", "statutorybody"],
            "party":   ["organization", "person", "financialinstitution", "government", "statutorybody"],
            "who":     ["organization", "person", "financialinstitution", "government", "statutorybody"],
            "court":   ["court"],
            "judge":   ["court", "person"],
            "amount":  ["demand", "amount", "transaction"],
            "demand":  ["demand"],
            "charges": ["demand"],
            "payment": ["transaction", "amount"],
            "order":   ["court", "legalproceeding"],
            "ruling":  ["court", "legalproceeding"],
            "case":    ["legalproceeding"],
            "petition":["legalproceeding"],
            "law":     ["regulation", "act"],
            "section": ["regulation"],
            "act":     ["regulation"],
            "asset":   ["asset"],
            "plot":    ["asset", "location"],
            "land":    ["asset", "location"],
            "loan":    ["transaction", "financialinstitution"],
            "bank":    ["financialinstitution"],
        }

        query_lower   = query.lower()
        matched_types: list[str] = []
        for keyword, types in type_keywords.items():
            if keyword in query_lower:
                matched_types.extend(types)

        if matched_types:
            type_seeds = []
            seen = set()
            for t in matched_types:
                for name in self._schema_type_map.get(t, [])[:3]:
                    if name not in seen:
                        type_seeds.append(name)
                        seen.add(name)
                if len(type_seeds) >= MAX_SEEDS:
                    break
            if type_seeds:
                return type_seeds[:MAX_SEEDS]

        # Tier 3 — high-degree fallback
        fallback = [n["name"] for n in self._high_degree_nodes[:MAX_SEEDS]]
        logger.info(f"No token/type match — using high-degree fallback: {fallback}")
        return fallback

    # ── Cypher Traversal ───────────────────────────────────────────────────────

    def _traverse(
        self,
        seed_names: list[str],
        hops: int,
        entity_filter: list[str] | None,
    ) -> list[dict]:
        """Expand subgraph from seed nodes, scoped to case_id."""
        driver = self._get_neo4j()
        with driver.session(database=settings.neo4j_database) as session:
            type_filter  = ""
            case_filter  = self._case_where("n")
            seed_filter  = self._case_where("seed")
            params       = self._case_params({"seeds": seed_names, "hops": hops})

            if entity_filter:
                type_filter  = "AND n.schemaType IN $types"
                params["types"] = entity_filter

            cypher = f"""
                MATCH (seed)
                WHERE seed.canonicalName IN $seeds {seed_filter}
                MATCH (seed)-[*1..{hops}]-(n)
                WHERE n.canonicalName IS NOT NULL
                  {case_filter}
                  {type_filter}
                WITH DISTINCT n
                OPTIONAL MATCH (n)-[r]-()
                WITH n, count(r) AS degree
                RETURN
                    n.canonicalName AS canonical_name,
                    n.text          AS text,
                    n.schemaType    AS schema_type,
                    n.sourcePDF     AS source_pdf,
                    n.sourcePage    AS source_page,
                    n.confidence    AS confidence,
                    degree
                ORDER BY degree DESC
                LIMIT 200
            """
            result = session.run(cypher, params)
            rows   = [dict(record) for record in result]

        seeds_data      = self._fetch_nodes_by_name(seed_names, entity_filter)
        seed_names_found = {r["canonical_name"] for r in rows}
        for s in seeds_data:
            if s["canonical_name"] not in seed_names_found:
                rows.append(s)

        return rows

    def _fetch_nodes_by_name(
        self,
        names: list[str],
        entity_filter: list[str] | None,
    ) -> list[dict]:
        """Fetch specific nodes by canonical name, scoped to case_id."""
        if not names:
            return []
        driver = self._get_neo4j()
        with driver.session(database=settings.neo4j_database) as session:
            case_filter  = self._case_where("n")
            type_filter  = "AND n.schemaType IN $types" if entity_filter else ""
            params       = self._case_params({"names": names})
            if entity_filter:
                params["types"] = entity_filter

            cypher = f"""
                MATCH (n)
                WHERE n.canonicalName IN $names
                  {case_filter}
                  {type_filter}
                OPTIONAL MATCH (n)-[r]-()
                WITH n, count(r) AS degree
                RETURN
                    n.canonicalName AS canonical_name,
                    n.text          AS text,
                    n.schemaType    AS schema_type,
                    n.sourcePDF     AS source_pdf,
                    n.sourcePage    AS source_page,
                    n.confidence    AS confidence,
                    degree
            """
            result = session.run(cypher, params)
            return [dict(record) for record in result]

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score_nodes(
        self,
        nodes: list[dict],
        query_tokens: set[str],
        faiss_similarities: dict[str, float],
    ) -> list[GraphNode]:
        import math

        alpha = settings.node_score_similarity
        beta  = settings.node_score_importance
        gamma = settings.node_score_confidence

        max_degree = max((n.get("degree", 1) for n in nodes), default=1)

        # Group by canonical_name — aggregate multiple entries from same node
        node_groups: dict[str, list[dict]] = {}
        for node_data in nodes:
            name = node_data.get("canonical_name") or ""
            node_groups.setdefault(name, []).append(node_data)

        scored: list[GraphNode] = []
        for name, entries in node_groups.items():
            primary    = entries[0]
            degree     = primary.get("degree") or 0
            confidence = primary.get("confidence") or 0.5

            chunk_scores = []
            for entry in entries:
                entry_conf = entry.get("confidence") or 0.5

                if faiss_similarities and name in faiss_similarities:
                    relevance = faiss_similarities[name]
                else:
                    node_tokens = (
                        _tokenize(name) |
                        _tokenize(entry.get("text") or "")
                    )
                    union     = query_tokens | node_tokens
                    relevance = (
                        len(query_tokens & node_tokens) / len(union)
                        if union else 0.0
                    )

                importance  = degree / max_degree if max_degree > 0 else 0.0
                chunk_score = (
                    alpha * relevance +
                    beta  * importance +
                    gamma * entry_conf
                )
                chunk_scores.append(chunk_score)

            # NodeScore = (1 / sqrt(N + 1)) * sum(chunk_scores)
            N          = len(chunk_scores)
            node_score = (1.0 / math.sqrt(N + 1)) * sum(chunk_scores)

            if faiss_similarities and name in faiss_similarities:
                display_relevance = faiss_similarities[name]
            else:
                node_tokens       = _tokenize(name) | _tokenize(primary.get("text") or "")
                union             = query_tokens | node_tokens
                display_relevance = (
                    len(query_tokens & node_tokens) / len(union)
                    if union else 0.0
                )

            scored.append(GraphNode(
                canonical_name       = name,
                text                 = primary.get("text") or name,
                schema_type          = primary.get("schema_type") or "Entity",
                source_pdf           = primary.get("source_pdf") or "",
                source_page          = primary.get("source_page") or 0,
                confidence           = float(confidence),
                degree               = degree,
                embedding_similarity = display_relevance,
                node_score           = round(node_score, 4),
            ))

        return scored

    # ── Subgraph ───────────────────────────────────────────────────────────────

    def _fetch_subgraph(self, node_names: list[str]) -> list[tuple[str, str, str]]:
        """Fetch relationships between retrieved nodes, scoped to case_id."""
        if not node_names:
            return []

        driver = self._get_neo4j()
        with driver.session(database=settings.neo4j_database) as session:
            case_filter = (
                "AND h.case_id = $case_id AND t.case_id = $case_id"
                if self._case_id else ""
            )
            result = session.run(f"""
                MATCH (h)-[r]->(t)
                WHERE h.canonicalName IN $names
                  AND t.canonicalName IN $names
                  {case_filter}
                RETURN
                    h.canonicalName AS head,
                    type(r)         AS relation,
                    t.canonicalName AS tail
            """, self._case_params({"names": node_names}))

            return [
                (record["head"], record["relation"], record["tail"])
                for record in result
            ]

    # ── FAISS (optional) ───────────────────────────────────────────────────────

    def _faiss_search(
        self, query: str, top_k: int
    ) -> tuple[dict[str, float], list[float]]:
        query_vec = self._text_encoder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        distances, indices = self._faiss_index.search(query_vec, top_k)
        similarities: dict[str, float] = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            name = self._idx_to_entity.get(int(idx))
            if name:
                similarities[name] = float(dist)

        return similarities, query_vec[0].tolist()


# ── Helpers ────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "and", "or", "not",
    "that", "this", "it", "its", "be", "has", "have", "had",
    "what", "who", "how", "why", "when", "which", "did", "does",
    "between", "about", "any", "all", "as",
}

def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    tokens = text.lower().replace("_", " ").split()
    return {
        t.strip(".,;:\"'()-")
        for t in tokens
        if len(t) > 2 and t not in _STOPWORDS
    }


# ── Standalone Test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    retriever = GraphRetriever()
    retriever.load(case_id="celir_llp_vs_midc")

    test_queries = [
        "Is MIDC justified in demanding ULC charges from CELIR?",
        "What is the relationship between Bafna Motors and Union Bank of India?",
        "What did the Supreme Court order on 21 September 2023?",
    ]

    query = test_queries[0] if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    print(f"\nQuery: '{query}'")
    print("-" * 60)

    results = retriever.search(query, top_k=10)

    print(f"\nTop nodes ({len(results.nodes)}):")
    for node in results.nodes:
        print(
            f"  [{node.schema_type}] {node.canonical_name} "
            f"| score={node.node_score} "
            f"| page={node.source_page} "
            f"| degree={node.degree}"
        )

    print(f"\nSubgraph triples ({len(results.subgraph_triples)}):")
    for head, rel, tail in results.subgraph_triples[:10]:
        print(f"  ({head}) --[{rel}]--> ({tail})")