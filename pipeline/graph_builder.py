"""
pipeline/graph_builder.py
--------------------------
Stage 3 of the ingestion pipeline.

Responsibilities:
- Take ExtractionResult (entities + relationships) from EntityExtractor
- Build Neo4j graph using parameterized Cypher (no injection risk)
- Track sourcePDF + sourcePage + confidence + case_id on every node and edge
- Deduplicate nodes using canonicalName + case_id as the MERGE key

Key design decisions:
    ✓ No hardcoded credentials (uses config.py)
    ✓ Parameterized Cypher (no string interpolation = no injection)
    ✓ Every node carries sourcePage + confidence + case_id
    ✓ Universal schema — no case-specific entity types hardcoded
    ✓ No relationship type whitelist — LLM confidence is the quality gate
    ✓ case_id partitioning — all cases share one Neo4j instance
      Queries filter by case_id so cases never bleed into each other
      Industry standard: tenant_id / case_id / project_id partitioning
      One instance supports thousands of cases with an index on case_id
    ✓ Cross-case intelligence enabled — query across case_id boundaries
      for entity matching across cases (e.g. "has MIDC appeared before?")
"""

from __future__ import annotations

import logging
from typing import Optional

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

from config import settings
from pipeline.entity_extractor import ExtractionResult, ExtractedEntity, ExtractedRelationship

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds and maintains the Neo4j knowledge graph.

    Usage:
        with GraphBuilder() as builder:
            builder.setup_constraints()
            builder.build_from_extraction(result, case_id="celir_llp_vs_midc")
    """

    def __init__(self):
        self._driver: Optional[Driver] = None

    def _get_driver(self) -> Driver:
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_username, settings.neo4j_password),
                )
                self._driver.verify_connectivity()
                logger.info("Connected to Neo4j successfully.")
            except AuthError:
                raise RuntimeError(
                    "Neo4j authentication failed. "
                    "Check NEO4J_USERNAME and NEO4J_PASSWORD in .env"
                )
            except ServiceUnavailable:
                raise RuntimeError(
                    f"Neo4j unavailable at {settings.neo4j_uri}. "
                    "Check NEO4J_URI in .env and that the instance is running."
                )
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self):
        self._get_driver()
        return self

    def __exit__(self, *args):
        self.close()

    # ── Schema Setup ───────────────────────────────────────────────────────────

    def setup_constraints(self):
        """
        Create Neo4j uniqueness constraints and indexes.
        Run once when setting up a new database.

        Uniqueness is now (canonicalName, case_id) — not just canonicalName.
        This allows the same entity name in different cases without collision.
        e.g. "midc" in celir_case and "midc" in another_case are separate nodes.
        """
        constraints = [
            # Uniqueness per (canonicalName, case_id) pair
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Person) REQUIRE (n.canonicalName, n.case_id) IS NODE KEY",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Organization) REQUIRE (n.canonicalName, n.case_id) IS NODE KEY",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Court) REQUIRE (n.canonicalName, n.case_id) IS NODE KEY",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:StatutoryBody) REQUIRE (n.canonicalName, n.case_id) IS NODE KEY",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:FinancialInstitution) REQUIRE (n.canonicalName, n.case_id) IS NODE KEY",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Asset) REQUIRE (n.canonicalName, n.case_id) IS NODE KEY",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:LegalProceeding) REQUIRE (n.canonicalName, n.case_id) IS NODE KEY",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Government) REQUIRE (n.canonicalName, n.case_id) IS NODE KEY",

            # case_id index — makes per-case queries fast even with millions of nodes
            "CREATE INDEX IF NOT EXISTS FOR (n:Person) ON (n.case_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Organization) ON (n.case_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Court) ON (n.case_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:StatutoryBody) ON (n.case_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:FinancialInstitution) ON (n.case_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Asset) ON (n.case_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:LegalProceeding) ON (n.case_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Government) ON (n.case_id)",

            # Additional useful indexes
            "CREATE INDEX IF NOT EXISTS FOR (n:Organization) ON (n.sourcePDF)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Person) ON (n.sourcePDF)",
        ]

        driver = self._get_driver()
        with driver.session(database=settings.neo4j_database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.debug(f"Constraint applied: {constraint[:60]}...")
                except Exception as e:
                    logger.warning(f"Constraint skipped (may already exist): {e}")

        logger.info("Neo4j constraints and indexes set up.")

    # ── Main Build Method ──────────────────────────────────────────────────────

    def build_from_extraction(self, result: ExtractionResult, case_id: str = ""):
        """
        Main entry point. Takes ExtractionResult and builds the graph.

        Args:
            result:  ExtractionResult from EntityExtractor
            case_id: Case identifier for partitioning — required for multi-case
        """
        driver = self._get_driver()

        if not case_id:
            # Fall back to pdf stem if case_id not provided
            case_id = result.source_pdf.replace(".pdf", "").replace(" ", "_").lower()
            logger.warning(
                f"No case_id provided — using '{case_id}' derived from PDF name. "
                "Pass case_id explicitly for proper partitioning."
            )

        logger.info(
            f"Building graph for {result.source_pdf} (case: {case_id}): "
            f"{len(result.unique_entities)} entities, "
            f"{len(result.relationships)} relationships"
        )

        entity_count = 0
        for entity in result.unique_entities:
            try:
                driver = self._get_driver()
                with driver.session(database=settings.neo4j_database) as session:
                    self._upsert_entity(session, entity, case_id)
                entity_count += 1
            except Exception as e:
                logger.warning(
                    f"Neo4j error on {entity.canonical_name}: {e} — reconnecting..."
                )
                self._driver = None
                import time; time.sleep(1)
                try:
                    driver = self._get_driver()
                    with driver.session(database=settings.neo4j_database) as session:
                        self._upsert_entity(session, entity, case_id)
                    entity_count += 1
                except Exception as e2:
                    logger.error(f"Failed after retry: {entity.canonical_name}: {e2}")

        rel_count = 0
        for rel in result.relationships:
            try:
                driver = self._get_driver()
                with driver.session(database=settings.neo4j_database) as session:
                    self._upsert_relationship(session, rel, case_id)
                rel_count += 1
            except Exception as e:
                logger.warning(f"Neo4j error on relationship: {e} — reconnecting...")
                self._driver = None
                import time; time.sleep(1)
                try:
                    driver = self._get_driver()
                    with driver.session(database=settings.neo4j_database) as session:
                        self._upsert_relationship(session, rel, case_id)
                    rel_count += 1
                except Exception as e2:
                    logger.error(
                        f"Failed after retry: {rel.from_entity} → {rel.to_entity}: {e2}"
                    )

        logger.info(
            f"Graph built: {entity_count} nodes, {rel_count} relationships "
            f"for {result.source_pdf} (case: {case_id})"
        )

    # ── Node Upsert ────────────────────────────────────────────────────────────

    def _upsert_entity(self, session, entity: ExtractedEntity, case_id: str):
        """
        MERGE entity into Neo4j using (canonicalName, case_id) as unique key.

        Same entity name in different cases creates separate nodes — no collision.
        Same entity name in the same case deduplicates correctly.
        """
        label = self._safe_label(entity.schema_type)

        query = f"""
        MERGE (n:{label} {{canonicalName: $canonicalName, case_id: $case_id}})
        ON CREATE SET
            n.text          = $text,
            n.schemaType    = $schemaType,
            n.glinerLabel   = $glinerLabel,
            n.confidence    = $confidence,
            n.sourcePDF     = $sourcePDF,
            n.sourcePage    = $sourcePage,
            n.context       = $context,
            n.case_id       = $case_id,
            n.sourceDocId   = $sourceDocId,
            n.sourceFilename = $sourceFilename,
            n.createdAt     = timestamp()
        ON MATCH SET
            n.confidence    = CASE
                                WHEN $confidence > n.confidence
                                THEN $confidence
                                ELSE n.confidence
                              END,
            n.updatedAt     = timestamp()
        """

        session.run(query, {
            "canonicalName": entity.canonical_name,
            "text":          entity.text,
            "schemaType":    entity.schema_type,
            "glinerLabel":   entity.gliner_label,
            "confidence":    entity.confidence,
            "sourcePDF":     entity.source_pdf,
            "sourcePage":    entity.source_page,
            "context":       entity.context,
            "case_id":       case_id,
            "sourceDocId":   getattr(entity, "source_doc_id", ""),
            "sourceFilename": getattr(entity, "source_filename", entity.source_pdf),
        })

    # ── Relationship Upsert ────────────────────────────────────────────────────

    def _upsert_relationship(self, session, rel: ExtractedRelationship, case_id: str):
        """
        MERGE relationship between two entities within the same case.

        Matches nodes by (canonicalName, case_id) so relationships never
        cross case boundaries accidentally.
        """
        rel_type = self._safe_rel_type(rel.relation_type)

        query = f"""
        MATCH (a {{canonicalName: $from_entity, case_id: $case_id}})
        MATCH (b {{canonicalName: $to_entity, case_id: $case_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        ON CREATE SET
            r.confidence  = $confidence,
            r.sourcePDF   = $sourcePDF,
            r.sourcePage  = $sourcePage,
            r.case_id     = $case_id,
            r.sourceDocId   = $sourceDocId,
            r.sourceFilename = $sourceFilename,
            r.createdAt   = timestamp()
        ON MATCH SET
            r.confidence  = CASE
                              WHEN $confidence > r.confidence
                              THEN $confidence
                              ELSE r.confidence
                            END
        """

        session.run(query, {
            "from_entity": rel.from_entity,
            "to_entity":   rel.to_entity,
            "confidence":  rel.confidence,
            "sourcePDF":   rel.source_pdf,
            "sourcePage":  rel.source_page,
            "case_id":     case_id,
            "sourceDocId": getattr(rel, "source_doc_id", ""),
            "sourceFilename": getattr(rel, "source_filename", rel.source_pdf),
        })

    # ── Safety Helpers ─────────────────────────────────────────────────────────

    _ALLOWED_LABELS = {
        "Person", "Organization", "Government", "Court",
        "StatutoryBody", "FinancialInstitution", "Asset",
        "Agreement", "Demand", "LegalProceeding", "Transaction",
        "Regulation", "Event", "Date", "Amount", "Identifier",
        "Location", "Entity",
    }

    def _safe_label(self, label: str) -> str:
        if label in self._ALLOWED_LABELS:
            return label
        logger.warning(f"Unknown schema label '{label}' — defaulting to Entity")
        return "Entity"

    def _safe_rel_type(self, rel_type: str) -> str:
        if not rel_type:
            return "RELATED_TO"
        sanitized = rel_type.upper().strip()
        sanitized = "".join(
            c if c.isalnum() or c == "_" else "_"
            for c in sanitized
        )
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        sanitized = sanitized.strip("_")
        if sanitized and sanitized[0].isdigit():
            sanitized = "REL_" + sanitized
        return sanitized or "RELATED_TO"

    # ── Utility Methods ────────────────────────────────────────────────────────

    def clear_case(self, case_id: str):
        """
        Delete all nodes and relationships for a specific case.
        Safe to call before re-ingesting a case.
        """
        driver = self._get_driver()
        with driver.session(database=settings.neo4j_database) as session:
            result = session.run(
                "MATCH (n {case_id: $case_id}) DETACH DELETE n RETURN count(n) AS deleted",
                case_id=case_id,
            )
            deleted = result.single()["deleted"]
        logger.info(f"Cleared {deleted} nodes for case '{case_id}'")

    def clear_graph(self):
        """Delete ALL nodes and relationships. USE WITH CAUTION."""
        driver = self._get_driver()
        with driver.session(database=settings.neo4j_database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.warning("Graph cleared — all nodes and relationships deleted.")

    def get_stats(self, case_id: str = None) -> dict:
        """
        Return graph statistics.

        Args:
            case_id: If provided, return stats for that case only.
                     If None, return stats for entire graph.
        """
        driver = self._get_driver()
        with driver.session(database=settings.neo4j_database) as session:
            if case_id:
                node_count = session.run(
                    "MATCH (n {case_id: $case_id}) RETURN count(n) AS c",
                    case_id=case_id,
                ).single()["c"]
                rel_count = session.run(
                    "MATCH ()-[r {case_id: $case_id}]->() RETURN count(r) AS c",
                    case_id=case_id,
                ).single()["c"]
                labels = session.run(
                    "MATCH (n {case_id: $case_id}) "
                    "RETURN DISTINCT labels(n) AS l, count(n) AS c",
                    case_id=case_id,
                ).data()
            else:
                node_count = session.run(
                    "MATCH (n) RETURN count(n) AS c"
                ).single()["c"]
                rel_count = session.run(
                    "MATCH ()-[r]->() RETURN count(r) AS c"
                ).single()["c"]
                labels = session.run(
                    "MATCH (n) RETURN DISTINCT labels(n) AS l, count(n) AS c"
                ).data()

        return {
            "case_id":             case_id or "all",
            "total_nodes":         node_count,
            "total_relationships": rel_count,
            "by_label":            labels,
        }

    def cross_case_search(self, canonical_name: str, exclude_case_id: str = None) -> list:
        """
        Search for an entity across ALL cases.
        Used for cross-case intelligence — 'has this entity appeared before?'

        Args:
            canonical_name:  Entity to search for
            exclude_case_id: Optionally exclude the current case from results
        """
        driver = self._get_driver()
        with driver.session(database=settings.neo4j_database) as session:
            query = """
            MATCH (n {canonicalName: $canonical_name})
            WHERE ($exclude_case IS NULL OR n.case_id <> $exclude_case)
            RETURN n.case_id AS case_id,
                   n.text AS text,
                   n.schemaType AS schema_type,
                   n.sourcePage AS source_page
            ORDER BY n.case_id
            """
            result = session.run(
                query,
                canonical_name=canonical_name,
                exclude_case=exclude_case_id,
            )
            return [dict(r) for r in result]


# ── Convenience Functions ──────────────────────────────────────────────────────

def build_graph(result: ExtractionResult, case_id: str = ""):
    """Convenience wrapper."""
    with GraphBuilder() as builder:
        builder.setup_constraints()
        builder.build_from_extraction(result, case_id=case_id)


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with GraphBuilder() as builder:
        stats = builder.get_stats()
        print(f"\nFull graph stats:")
        print(f"  Nodes:         {stats['total_nodes']}")
        print(f"  Relationships: {stats['total_relationships']}")
        print(f"  By label:      {stats['by_label']}")