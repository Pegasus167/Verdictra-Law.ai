"""
pipeline/relationship_extractor.py
------------------------------------
LLM-based typed relationship extraction with async concurrent calls.

How it works:
    1. GLiNER extracts entities (unchanged)
    2. For each page, build entity pairs from co-occurring entities
    3. Find the sentence(s) containing both entities
    4. Fire ALL batches concurrently to GPT-4o-mini via asyncio
    5. LLM returns typed relationship or NULL per pair
    6. High confidence (>= 0.70) → typed relationship in Neo4j
    7. Low confidence or NULL → skipped entirely

Speed improvement:
    Sequential (old): 5 batches × 1.5s = 7.5s per page
    Concurrent (new): all 5 batches fire at once = 1.5s per page
    ~5x faster on relationship extraction stage
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)


# ── Relationship vocabulary ────────────────────────────────────────────────────

TYPED_RELATIONSHIPS = {
    "Legal proceedings": [
        "PETITIONS_AGAINST", "FILED_BY", "CHALLENGED_BY",
        "ORDERED_BY", "GOVERNED_BY", "REPRESENTED_BY",
        "APPOINTED_BY", "PARTY_TO",
    ],
    "Ownership and transfer": [
        "LEASED_TO", "TRANSFERRED_TO", "ALLOTTED_TO",
        "SUBLET_TO", "OWNED_BY", "MORTGAGED_WITH",
        "AUCTIONED_BY", "SOLD_TO",
    ],
    "Financial": [
        "DEMANDED_FROM", "ISSUED_TO", "PAID_BY",
        "LOANED_TO", "ISSUED_LOAN_TO", "GUARANTEED_BY",
        "SECURED_BY",
    ],
    "Corporate": [
        "SUBSIDIARY_OF", "DIRECTOR_OF", "PARTNER_OF",
        "MEMBER_OF", "MANAGED_BY",
    ],
    "General": [
        "LOCATED_AT", "REFERENCED_IN", "RELATED_TO",
    ],
}

MIN_RELATIONSHIP_CONFIDENCE = 0.70
MAX_PAIRS_PER_CALL = 10

MEANINGFUL_TYPES = {
    "Person", "Organization", "Court", "StatutoryBody",
    "FinancialInstitution", "Asset", "LegalProceeding",
    "Demand", "Agreement", "Transaction", "Government",
    "Regulation", "Event",
}


# ── Sentence finder ────────────────────────────────────────────────────────────

def _find_shared_sentences(
    text: str,
    entity1_text: str,
    entity2_text: str,
    window: int = 2,
) -> str:
    """Find sentences containing both entity mentions."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    e1_lower = entity1_text.lower()
    e2_lower = entity2_text.lower()

    relevant = []
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        has_e1 = e1_lower in sent_lower
        has_e2 = e2_lower in sent_lower

        if has_e1 and has_e2:
            start = max(0, i - 1)
            end = min(len(sentences), i + 2)
            relevant.extend(sentences[start:end])
            break
        elif has_e1 or has_e2:
            relevant.append(sent)

    if not relevant:
        return text[:300]

    return " ".join(relevant[:window * 2])


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(pair_descriptions: list[dict]) -> str:
    rel_list = "\n".join(
        f"  {category}: {', '.join(rels)}"
        for category, rels in TYPED_RELATIONSHIPS.items()
    )
    pairs_json = json.dumps(pair_descriptions, indent=2)

    return f"""You are an expert legal document analyst extracting typed relationships.

For each entity pair below, determine if there is a direct semantic relationship
between them based ONLY on the provided context sentences.

Available relationship types:
{rel_list}

Rules:
- Only assign a relationship if the context CLEARLY states it
- The relationship must be DIRECTIONAL: Entity A → relationship → Entity B
- If no clear relationship exists, set relationship to NULL
- Confidence should reflect how clearly the context states the relationship
- Do not infer relationships not stated in the context

Entity pairs to analyze:
{pairs_json}

Respond with this exact JSON format:
{{
  "pairs": [
    {{
      "pair_id": 1,
      "relationship": "PETITIONS_AGAINST",
      "direction": "A_TO_B",
      "confidence": 0.95,
      "evidence": "exact quote from context supporting this relationship"
    }},
    {{
      "pair_id": 2,
      "relationship": null,
      "direction": null,
      "confidence": 0.0,
      "evidence": ""
    }}
  ]
}}"""


# ── Result parser ──────────────────────────────────────────────────────────────

def _parse_result(
    result: dict,
    entity_pairs: list[tuple],
    source_pdf: str,
    source_page: int,
) -> list[dict]:
    """Parse LLM JSON result into relationship dicts."""
    relationships = []

    for item in result.get("pairs", []):
        pair_id = item.get("pair_id", 0) - 1
        relationship = item.get("relationship")
        confidence = float(item.get("confidence", 0.0))
        direction = item.get("direction", "A_TO_B")
        evidence = item.get("evidence", "")

        if not relationship or confidence < MIN_RELATIONSHIP_CONFIDENCE:
            continue

        if pair_id < 0 or pair_id >= len(entity_pairs):
            continue

        e1, e2 = entity_pairs[pair_id]

        if direction == "B_TO_A":
            from_entity = e2.canonical_name
            to_entity = e1.canonical_name
        else:
            from_entity = e1.canonical_name
            to_entity = e2.canonical_name

        relationships.append({
            "from_entity":   from_entity,
            "to_entity":     to_entity,
            "relation_type": relationship,
            "confidence":    round(confidence, 4),
            "source_pdf":    source_pdf,
            "source_page":   source_page,
            "evidence":      evidence[:200],
        })

    return relationships


# ── Extractor class ────────────────────────────────────────────────────────────

class RelationshipExtractor:
    """
    Extracts typed relationships between entity pairs.

    Has two modes:
        extract_relationships()       — synchronous, for simple use
        extract_relationships_async() — async, used by extract_relationships_for_page_async()
    """

    def __init__(self):
        self._client = None
        self._async_client = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            from config import settings
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def _get_async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            from config import settings
            self._async_client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._async_client

    # ── Synchronous (used for single calls / testing) ──────────────────────────

    def extract_relationships(
        self,
        entity_pairs: list[tuple],
        page_text: str,
        source_pdf: str,
        source_page: int,
    ) -> list[dict]:
        """Synchronous relationship extraction for a single batch."""
        if not entity_pairs:
            return []

        pair_descriptions = []
        for i, (e1, e2) in enumerate(entity_pairs[:MAX_PAIRS_PER_CALL]):
            context = _find_shared_sentences(page_text, e1.text, e2.text)
            pair_descriptions.append({
                "pair_id": i + 1,
                "entity_a": {
                    "name": e1.text,
                    "type": e1.schema_type,
                    "canonical": e1.canonical_name,
                },
                "entity_b": {
                    "name": e2.text,
                    "type": e2.schema_type,
                    "canonical": e2.canonical_name,
                },
                "context": context,
            })

        prompt = _build_prompt(pair_descriptions)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=800,
            )
            result = json.loads(response.choices[0].message.content)
            return _parse_result(result, entity_pairs, source_pdf, source_page)

        except Exception as e:
            logger.error(
                f"Relationship extraction failed on page {source_page}: {e}"
            )
            return []

    # ── Async (used by extract_relationships_for_page_async) ──────────────────

    async def extract_relationships_async(
        self,
        entity_pairs: list[tuple],
        page_text: str,
        source_pdf: str,
        source_page: int,
    ) -> list[dict]:
        """Async relationship extraction — non-blocking, for concurrent calls."""
        if not entity_pairs:
            return []

        pair_descriptions = []
        for i, (e1, e2) in enumerate(entity_pairs[:MAX_PAIRS_PER_CALL]):
            context = _find_shared_sentences(page_text, e1.text, e2.text)
            pair_descriptions.append({
                "pair_id": i + 1,
                "entity_a": {
                    "name": e1.text,
                    "type": e1.schema_type,
                    "canonical": e1.canonical_name,
                },
                "entity_b": {
                    "name": e2.text,
                    "type": e2.schema_type,
                    "canonical": e2.canonical_name,
                },
                "context": context,
            })

        prompt = _build_prompt(pair_descriptions)

        try:
            client = self._get_async_client()
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=800,
            )
            result = json.loads(response.choices[0].message.content)
            return _parse_result(result, entity_pairs, source_pdf, source_page)

        except Exception as e:
            logger.error(
                f"Async relationship extraction failed on page {source_page}: {e}"
            )
            return []


# ── Page-level processors ──────────────────────────────────────────────────────

def extract_relationships_for_page(
    entities: list,
    page_text: str,
    source_pdf: str,
    source_page: int,
    extractor: RelationshipExtractor = None,
) -> list:
    """
    Synchronous page processor.
    Kept for backward compatibility and simple testing.
    For production use extract_relationships_for_page_async().
    """
    from pipeline.entity_extractor import ExtractedRelationship

    if extractor is None:
        extractor = RelationshipExtractor()

    meaningful = [e for e in entities if e.schema_type in MEANINGFUL_TYPES]

    pairs = []
    seen = set()
    for i, e1 in enumerate(meaningful):
        for e2 in meaningful[i + 1:]:
            if e1.canonical_name == e2.canonical_name:
                continue
            key = tuple(sorted([e1.canonical_name, e2.canonical_name]))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((e1, e2))

    if not pairs:
        return []

    all_relationships = []
    for i in range(0, len(pairs), MAX_PAIRS_PER_CALL):
        batch = pairs[i: i + MAX_PAIRS_PER_CALL]
        raw = extractor.extract_relationships(
            batch, page_text, source_pdf, source_page
        )
        for r in raw:
            all_relationships.append(ExtractedRelationship(
                from_entity=r["from_entity"],
                to_entity=r["to_entity"],
                relation_type=r["relation_type"],
                source_pdf=r["source_pdf"],
                source_page=r["source_page"],
                confidence=r["confidence"],
            ))

    if all_relationships:
        logger.info(
            f"  Page {source_page}: {len(pairs)} pairs → "
            f"{len(all_relationships)} typed relationships"
        )

    return all_relationships


async def extract_relationships_for_page_async(
    entities: list,
    page_text: str,
    source_pdf: str,
    source_page: int,
    extractor: RelationshipExtractor = None,
) -> list:
    """
    Async concurrent page processor.

    Fires all batch LLM calls simultaneously using asyncio.gather().
    A page with 45 pairs = 5 batches fired at once instead of sequentially.
    ~5x faster than the synchronous version.
    """
    from pipeline.entity_extractor import ExtractedRelationship

    if extractor is None:
        extractor = RelationshipExtractor()

    meaningful = [e for e in entities if e.schema_type in MEANINGFUL_TYPES]

    pairs = []
    seen = set()
    for i, e1 in enumerate(meaningful):
        for e2 in meaningful[i + 1:]:
            if e1.canonical_name == e2.canonical_name:
                continue
            key = tuple(sorted([e1.canonical_name, e2.canonical_name]))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((e1, e2))

    if not pairs:
        return []

    # Build all batch tasks
    tasks = []
    for i in range(0, len(pairs), MAX_PAIRS_PER_CALL):
        batch = pairs[i: i + MAX_PAIRS_PER_CALL]
        tasks.append(
            extractor.extract_relationships_async(
                batch, page_text, source_pdf, source_page
            )
        )

    # Fire all batches concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_relationships = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(
                f"Async batch failed on page {source_page}: {result}"
            )
            continue
        for r in result:
            all_relationships.append(ExtractedRelationship(
                from_entity=r["from_entity"],
                to_entity=r["to_entity"],
                relation_type=r["relation_type"],
                source_pdf=r["source_pdf"],
                source_page=r["source_page"],
                confidence=r["confidence"],
            ))

    if all_relationships:
        logger.info(
            f"  Page {source_page}: {len(pairs)} pairs → "
            f"{len(all_relationships)} typed relationships"
        )

    return all_relationships