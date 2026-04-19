"""
pipeline/relationship_upgrader.py
-----------------------------------
Generates typed relationships between entities using LLM reasoning.

Replaces the old CO_OCCURS_WITH brute-force approach with:
    1. Smart candidate pair filtering (confidence + type + context)
    2. Batched LLM calls (10 pairs per call — cost efficient)
    3. Whitelisted relationship types — LLM picks or says NONE
    4. Creates typed Neo4j relationships directly

Cost estimate for ~2000 entity graph:
    ~800 candidate pairs → 80 batched LLM calls → ~$0.16 total

Run after:
    - ingestion.py (entities in Neo4j)
    - entity_resolver.py (duplicates merged)
    - Delete all CO_OCCURS_WITH first:
        MATCH ()-[r:CO_OCCURS_WITH]->() DELETE r

Usage:
    poetry run python pipeline/relationship_upgrader.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


# ── Whitelisted relationship types ─────────────────────────────────────────────
# LLM must pick from this list — cannot invent new types

ALLOWED_RELATIONSHIPS = {
    # Legal proceedings
    "FILED_BY",
    "PETITIONS_AGAINST",
    "RULED_FOR",
    "RULED_AGAINST",
    "DIRECTED",
    "INVOLVES",
    "CHALLENGED_BY",
    "APPEALED_BY",

    # Financial
    "MORTGAGED_WITH",
    "HOLDS_CHARGE_OVER",
    "ISSUED_LOAN_TO",
    "AUCTIONED_BY",
    "PAID_TO",
    "OWES_TO",

    # Organizational
    "DIRECTOR_OF",
    "AFFILIATED_WITH",
    "OVERSEES",
    "HAS_OFFICER",
    "ADVOCATES_FOR",
    "REPRESENTS",
    "SUBSIDIARY_OF",

    # Property / Asset
    "LEASED_TO",
    "TRANSFERRED_TO",
    "SUBLET_TO",
    "OWNS",
    "CURRENT_LESSEE_OF",

    # Documents / Demands
    "ISSUED_BY",
    "ISSUED_TO",
    "REFERENCES",
    "GOVERNED_BY",

    # Fallbacks
    "RELATED_TO",   # Weak but typed — use when relationship exists but unclear
    "NONE",         # No meaningful relationship — skip entirely
}

# Entity types worth building relationships between
# Dates, Amounts, Identifiers are attributes — not relationship nodes
MEANINGFUL_TYPES = {
    "Person", "Organization", "Government", "Court",
    "StatutoryBody", "FinancialInstitution", "Asset",
    "Agreement", "Demand", "LegalProceeding", "Transaction",
    "Regulation", "Event",
}

# Type pairs that almost never have meaningful relationships — skip
SKIP_TYPE_PAIRS = {
    ("Location", "Location"),
    ("Date", "Date"),
    ("Amount", "Amount"),
    ("Identifier", "Identifier"),
    ("Location", "Date"),
    ("Date", "Amount"),
}

# Batch size — pairs per LLM call
BATCH_SIZE = 10

# Minimum entity confidence to be considered for relationship generation
MIN_CONFIDENCE = 0.5


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class EntityPair:
    """A candidate pair of entities for relationship extraction."""
    head_name: str
    head_type: str
    head_text: str
    tail_name: str
    tail_type: str
    tail_text: str
    source_page: int
    context: str            # Combined context from both entities
    source_pdf: str


@dataclass
class ExtractedRelationship:
    """A relationship extracted by the LLM."""
    head_name: str
    tail_name: str
    relation_type: str
    confidence: float
    reason: str
    source_page: int
    source_pdf: str


# ── Main Upgrader ──────────────────────────────────────────────────────────────

class RelationshipUpgrader:
    """
    Generates typed relationships between entities using batched LLM calls.

    Usage:
        upgrader = RelationshipUpgrader()
        upgrader.run(extraction_json_path)
    """

    def __init__(self):
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
            )
        return self._driver

    # ── Step 1: Load entities from extraction JSON ─────────────────────────────

    def load_entities_by_page(
        self, extraction_json_path: str | Path
    ) -> dict[int, list[dict]]:
        """
        Load entities grouped by page from extraction JSON.
        Returns {page_number: [entity_dict, ...]}
        """
        with open(extraction_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pages: dict[int, list[dict]] = {}
        for entity in data.get("entities", []):
            # Filter by confidence and meaningful type
            if entity.get("confidence", 0) < MIN_CONFIDENCE:
                continue
            if entity.get("schema_type") not in MEANINGFUL_TYPES:
                continue

            page = entity.get("source_page", 0)
            if page > 0:
                pages.setdefault(page, []).append(entity)

        logger.info(
            f"Loaded entities from {len(pages)} pages "
            f"(min confidence: {MIN_CONFIDENCE})"
        )
        return pages

    # ── Step 2: Generate candidate pairs ──────────────────────────────────────

    def generate_candidate_pairs(
        self, entities_by_page: dict[int, list[dict]]
    ) -> list[EntityPair]:
        """
        Generate candidate entity pairs from same-page entities.
        Applies filtering to reduce noise before LLM calls.
        """
        pairs = []
        seen = set()  # Avoid duplicate pairs

        for page, entities in entities_by_page.items():
            if len(entities) < 2:
                continue

            # Generate all pairs on this page
            for e1, e2 in combinations(entities, 2):
                # Skip same entity
                if e1["canonical_name"] == e2["canonical_name"]:
                    continue

                # Skip already seen pairs (order-independent)
                pair_key = tuple(sorted([e1["canonical_name"], e2["canonical_name"]]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                # Skip meaningless type pairs
                type_pair = tuple(sorted([e1["schema_type"], e2["schema_type"]]))
                if type_pair in SKIP_TYPE_PAIRS:
                    continue

                # Build combined context
                ctx1 = e1.get("context", "")
                ctx2 = e2.get("context", "")
                combined_context = f"{ctx1} ... {ctx2}" if ctx1 != ctx2 else ctx1

                pairs.append(EntityPair(
                    head_name=e1["canonical_name"],
                    head_type=e1["schema_type"],
                    head_text=e1["text"],
                    tail_name=e2["canonical_name"],
                    tail_type=e2["schema_type"],
                    tail_text=e2["text"],
                    source_page=page,
                    context=combined_context[:400],
                    source_pdf=e1.get("source_pdf", ""),
                ))

        logger.info(f"Generated {len(pairs)} candidate pairs for LLM scoring")
        return pairs

    # ── Step 3: LLM scoring in batches ────────────────────────────────────────

    def score_pairs_with_llm(
        self, pairs: list[EntityPair]
    ) -> list[ExtractedRelationship]:
        """
        Send pairs to LLM in batches of BATCH_SIZE.
        Returns extracted typed relationships.
        """
        relationships = []
        total_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info(
            f"Scoring {len(pairs)} pairs in {total_batches} batches "
            f"of {BATCH_SIZE}..."
        )

        for batch_idx in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[batch_idx : batch_idx + BATCH_SIZE]
            batch_num = batch_idx // BATCH_SIZE + 1

            logger.info(f"  Batch {batch_num}/{total_batches}...")

            try:
                batch_results = self._call_llm_batch(batch)
                relationships.extend(batch_results)
            except Exception as e:
                logger.error(f"  Batch {batch_num} failed: {e}")
                continue

        # Filter out NONE relationships
        typed = [r for r in relationships if r.relation_type != "NONE"]
        logger.info(
            f"LLM extracted {len(typed)} typed relationships "
            f"(from {len(relationships)} total, "
            f"{len(relationships) - len(typed)} discarded as NONE)"
        )
        return typed

    def _call_llm_batch(self, batch: list[EntityPair]) -> list[ExtractedRelationship]:
        """Call LLM for a single batch of pairs."""

        # Build pair descriptions
        pair_descriptions = []
        for i, pair in enumerate(batch):
            pair_descriptions.append(
                f"Pair {i+1}:\n"
                f"  Entity A: \"{pair.head_text}\" [{pair.head_type}]\n"
                f"  Entity B: \"{pair.tail_text}\" [{pair.tail_type}]\n"
                f"  Page: {pair.source_page}\n"
                f"  Context: \"{pair.context}\""
            )

        allowed_str = ", ".join(sorted(ALLOWED_RELATIONSHIPS))

        prompt = f"""You are a legal document relationship extraction expert.

For each entity pair below, determine if there is a meaningful legal or 
organizational relationship between them based on the context provided.

Allowed relationship types (pick EXACTLY one):
{allowed_str}

Rules:
- Direction matters: A → B means A does the action TO/WITH B
- Use NONE if there is no clear relationship in the context
- Use RELATED_TO only if a relationship clearly exists but type is unclear
- Base your answer ONLY on the provided context — do not invent relationships
- A Person DIRECTOR_OF an Organization, not the reverse
- A Court RULED_FOR/RULED_AGAINST a party, not the reverse

{chr(10).join(pair_descriptions)}

Respond in this exact JSON format:
{{
  "relationships": [
    {{
      "pair_number": 1,
      "relation_type": "PETITIONS_AGAINST",
      "direction": "A_TO_B",
      "confidence": 0.92,
      "reason": "Context shows A filed writ petition challenging B's demand"
    }},
    ...
  ]
}}

Include one entry per pair. Use NONE if no relationship exists."""

        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1000,
        )

        data = json.loads(response.choices[0].message.content)
        results = []

        for item in data.get("relationships", []):
            pair_idx = item.get("pair_number", 1) - 1
            if pair_idx >= len(batch):
                continue

            pair = batch[pair_idx]
            rel_type = item.get("relation_type", "NONE")

            # Validate relationship type
            if rel_type not in ALLOWED_RELATIONSHIPS:
                logger.warning(
                    f"LLM returned unknown relationship '{rel_type}' — "
                    f"defaulting to RELATED_TO"
                )
                rel_type = "RELATED_TO"

            # Handle direction
            direction = item.get("direction", "A_TO_B")
            if direction == "B_TO_A":
                head, tail = pair.tail_name, pair.head_name
            else:
                head, tail = pair.head_name, pair.tail_name

            results.append(ExtractedRelationship(
                head_name=head,
                tail_name=tail,
                relation_type=rel_type,
                confidence=float(item.get("confidence", 0.7)),
                reason=item.get("reason", ""),
                source_page=pair.source_page,
                source_pdf=pair.source_pdf,
            ))

        return results

    # ── Step 4: Write to Neo4j ─────────────────────────────────────────────────

    def write_relationships_to_neo4j(
        self, relationships: list[ExtractedRelationship]
    ):
        """
        Create typed relationships in Neo4j.
        Uses MERGE to avoid duplicates if run multiple times.
        """
        driver = self._get_driver()
        created = 0
        failed = 0

        with driver.session(database=settings.neo4j_database) as session:
            for rel in relationships:
                # Whitelist check — safety guard
                rel_type = rel.relation_type
                if rel_type not in ALLOWED_RELATIONSHIPS or rel_type == "NONE":
                    continue

                try:
                    session.run(f"""
                        MATCH (h {{canonicalName: $head}})
                        MATCH (t {{canonicalName: $tail}})
                        MERGE (h)-[r:{rel_type}]->(t)
                        ON CREATE SET
                            r.confidence  = $confidence,
                            r.reason      = $reason,
                            r.sourcePage  = $source_page,
                            r.sourcePDF   = $source_pdf,
                            r.generatedBy = 'llm_relationship_upgrader',
                            r.createdAt   = timestamp()
                    """, {
                        "head": rel.head_name,
                        "tail": rel.tail_name,
                        "confidence": rel.confidence,
                        "reason": rel.reason,
                        "source_page": rel.source_page,
                        "source_pdf": rel.source_pdf,
                    })
                    created += 1
                except Exception as e:
                    logger.error(
                        f"Failed to create {rel.head_name} "
                        f"-[{rel_type}]-> {rel.tail_name}: {e}"
                    )
                    failed += 1

        logger.info(
            f"Neo4j: {created} relationships created, {failed} failed"
        )
        return created

    # ── Step 5: Save results log ───────────────────────────────────────────────

    def save_results_log(
        self,
        relationships: list[ExtractedRelationship],
        log_path: str | Path = "data/relationship_upgrade_log.json",
    ):
        """Save all extracted relationships to a JSON log for inspection."""
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        log = {
            "total_relationships": len(relationships),
            "by_type": {},
            "relationships": [],
        }

        for rel in relationships:
            log["by_type"][rel.relation_type] = (
                log["by_type"].get(rel.relation_type, 0) + 1
            )
            log["relationships"].append({
                "head": rel.head_name,
                "relation": rel.relation_type,
                "tail": rel.tail_name,
                "confidence": rel.confidence,
                "reason": rel.reason,
                "page": rel.source_page,
            })

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

        logger.info(f"Results log saved → {log_path}")
        logger.info("Relationship type breakdown:")
        for rel_type, count in sorted(
            log["by_type"].items(), key=lambda x: -x[1]
        ):
            logger.info(f"  {rel_type}: {count}")

    # ── Main run ───────────────────────────────────────────────────────────────

    def run(self, extraction_json_path: str | Path):
        """Full relationship upgrade pipeline."""

        logger.info("Step 1: Loading entities from extraction JSON...")
        entities_by_page = self.load_entities_by_page(extraction_json_path)

        logger.info("Step 2: Generating candidate pairs...")
        pairs = self.generate_candidate_pairs(entities_by_page)

        if not pairs:
            logger.warning("No candidate pairs found. Check entity confidence levels.")
            return

        logger.info("Step 3: LLM scoring in batches...")
        relationships = self.score_pairs_with_llm(pairs)

        if not relationships:
            logger.warning("No typed relationships extracted.")
            return

        logger.info("Step 4: Writing to Neo4j...")
        created = self.write_relationships_to_neo4j(relationships)

        logger.info("Step 5: Saving results log...")
        self.save_results_log(relationships)

        logger.info(
            f"\n✓ Relationship upgrade complete."
            f"\n  Candidate pairs:       {len(pairs)}"
            f"\n  Typed relationships:   {len(relationships)}"
            f"\n  Created in Neo4j:      {created}"
            f"\n  Log: data/relationship_upgrade_log.json"
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    extraction_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/json_output/celir_case_extraction.json"
    )

    if not Path(extraction_path).exists():
        print(f"Extraction JSON not found: {extraction_path}")
        print("Run ingestion.py first.")
        sys.exit(1)

    upgrader = RelationshipUpgrader()
    upgrader.run(extraction_path)