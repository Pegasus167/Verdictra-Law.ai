"""
pipeline/entity_resolver.py
----------------------------
Human-in-the-Loop Entity Resolution System.

Flow:
    1. Load extracted entities from JSON
    2. Group by schema type
    3. Find candidate merge groups using embedding similarity
    4. Score each group with LLM (GPT-4o-mini simple, GPT-4o complex)
    5. Split into: AUTO_MERGE / NEEDS_REVIEW / AUTO_KEEP
    6. Save resolution state to JSON for the web UI
    7. After human review, apply decisions to Neo4j

A group is COMPLEX if:
    - More than 3 candidates, OR
    - Any candidate has confidence between 0.35–0.65
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
CANDIDATE_SIMILARITY_THRESHOLD = 0.65
AUTO_MERGE_CONFIDENCE = 0.85
AUTO_KEEP_CONFIDENCE = 0.35
COMPLEX_GROUP_SIZE = 3


# ── Data Structures ────────────────────────────────────────────────────────────

class Resolution(str, Enum):
    AUTO_MERGE = "AUTO_MERGE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    AUTO_KEEP = "AUTO_KEEP"


@dataclass
class CandidateEntity:
    canonical_name: str
    text: str
    schema_type: str
    source_pdf: str
    source_page: int
    source_line: int
    context: str
    confidence: float
    llm_merge_confidence: float = 0.0
    llm_vote: str = ""
    llm_reason: str = ""


@dataclass
class ResolutionGroup:
    group_id: str
    schema_type: str
    candidates: list[CandidateEntity]
    resolution: Resolution = Resolution.NEEDS_REVIEW
    canonical_name: str = ""
    is_complex: bool = False
    llm_model_used: str = ""
    human_decision: Optional[str] = None
    human_selected: list[str] = field(default_factory=list)
    disagreement: bool = False
    decided_at: str = ""


# ── Main Resolver ──────────────────────────────────────────────────────────────

class EntityResolver:
    """
    Core entity resolution engine.

    Usage:
        resolver = EntityResolver()

        # Full pipeline in one call (used by app.py background ingestion)
        resolver.resolve(extraction_json_path, output_path)

        # Or step by step
        entities = resolver.load_entities(extraction_json_path)
        groups = resolver.find_candidate_groups(entities)
        groups = resolver.score_groups_with_llm(groups)
        resolver.save_resolution_state(groups, output_path)
    """

    def __init__(self):
        self._embedding_model = None
        self._openai_client = None

    # ── Full pipeline in one call ──────────────────────────────────────────────

    def resolve(
        self,
        extraction_json_path: str | Path,
        output_path: str | Path,
    ) -> list[ResolutionGroup]:
        """
        Full resolution pipeline in one call.
        Used by app.py background ingestion after entity extraction completes.

        Args:
            extraction_json_path: Path to cases/{case_id}/extraction.json
            output_path:          Path to cases/{case_id}/resolution_state.json

        Returns:
            List of ResolutionGroup objects (also saved to output_path)
        """
        logger.info(f"[Resolver] Loading entities from {extraction_json_path}...")
        entities = self.load_entities(extraction_json_path)

        logger.info(f"[Resolver] Finding candidate groups...")
        groups = self.find_candidate_groups(entities)

        logger.info(f"[Resolver] LLM scoring {len(groups)} groups...")
        groups = self.score_groups_with_llm(groups)

        logger.info(f"[Resolver] Saving resolution state to {output_path}...")
        self.save_resolution_state(groups, output_path)

        return groups

    # ── Model loading ──────────────────────────────────────────────────────────

    def _get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info("Loading sentence transformer for entity similarity...")
            self._embedding_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._embedding_model

    def _get_openai(self) -> OpenAI:
        if self._openai_client is None:
            from config import settings
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    # ── Step 1: Load entities ──────────────────────────────────────────────────

    def load_entities(self, json_path: str | Path) -> list[CandidateEntity]:
        """Load extracted entities from extraction.json."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entities = []
        for item in data.get("entities", []):
            line_num = self._estimate_line(item.get("context", ""), item.get("source_page", 0))
            entities.append(CandidateEntity(
                canonical_name=item["canonical_name"],
                text=item["text"],
                schema_type=item["schema_type"],
                source_pdf=item["source_pdf"],
                source_page=item["source_page"],
                source_line=line_num,
                context=item.get("context", ""),
                confidence=item.get("confidence", 0.5),
            ))

        logger.info(f"Loaded {len(entities)} entities from {json_path}")
        return entities

    def _estimate_line(self, context: str, page: int) -> int:
        return max(1, context.count("\n") + 1)

    # ── Step 2: Find candidate groups ─────────────────────────────────────────

    def find_candidate_groups(
        self,
        entities: list[CandidateEntity],
    ) -> list[ResolutionGroup]:
        """Group similar entities using embedding similarity + Union-Find."""
        model = self._get_embedding_model()
        groups = []
        group_counter = 0

        types = set(e.schema_type for e in entities)

        for schema_type in types:
            type_entities = [e for e in entities if e.schema_type == schema_type]
            if len(type_entities) < 2:
                continue

            names = [e.canonical_name.replace("_", " ") for e in type_entities]
            embeddings = model.encode(names, normalize_embeddings=True)
            similarity_matrix = np.dot(embeddings, embeddings.T)

            clusters = self._cluster_similar(
                type_entities, similarity_matrix, CANDIDATE_SIMILARITY_THRESHOLD
            )

            for cluster in clusters:
                if len(cluster) < 2:
                    continue

                group_counter += 1
                is_complex = len(cluster) > COMPLEX_GROUP_SIZE

                group = ResolutionGroup(
                    group_id=f"group_{group_counter:04d}",
                    schema_type=schema_type,
                    candidates=cluster,
                    is_complex=is_complex,
                    canonical_name=self._pick_canonical(cluster),
                )
                groups.append(group)

        logger.info(
            f"Found {len(groups)} candidate groups "
            f"({sum(1 for g in groups if g.is_complex)} complex)"
        )
        return groups

    def _cluster_similar(
        self,
        entities: list[CandidateEntity],
        similarity_matrix: np.ndarray,
        threshold: float,
    ) -> list[list[CandidateEntity]]:
        """Union-Find clustering."""
        n = len(entities)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] >= threshold:
                    union(i, j)

        clusters: dict[int, list[CandidateEntity]] = {}
        for i, entity in enumerate(entities):
            root = find(i)
            clusters.setdefault(root, []).append(entity)

        return list(clusters.values())

    def _pick_canonical(self, candidates: list[CandidateEntity]) -> str:
        """Pick best canonical name — shortest + highest confidence."""
        sorted_candidates = sorted(
            candidates,
            key=lambda e: (len(e.canonical_name), -e.confidence)
        )
        return sorted_candidates[0].canonical_name

    # ── Step 3: LLM scoring ────────────────────────────────────────────────────

    def score_groups_with_llm(
        self,
        groups: list[ResolutionGroup],
    ) -> list[ResolutionGroup]:
        """Score each group with GPT-4o-mini (simple) or GPT-4o (complex)."""
        for i, group in enumerate(groups):
            model = "gpt-4o" if group.is_complex else "gpt-4o-mini"
            logger.info(
                f"Scoring group {i+1}/{len(groups)} "
                f"[{group.schema_type}] with {model}..."
            )
            self._score_group(group, model)
        return groups

    def _score_group(self, group: ResolutionGroup, model: str):
        """Ask LLM to evaluate candidates in this group."""
        client = self._get_openai()
        group.llm_model_used = model

        candidate_descriptions = []
        for i, c in enumerate(group.candidates):
            candidate_descriptions.append(
                f"Candidate {i+1}:\n"
                f"  Name: \"{c.text}\"\n"
                f"  Type: {c.schema_type}\n"
                f"  Found at: Page {c.source_page}, Line {c.source_line}\n"
                f"  Context: \"{c.context[:150]}\""
            )

        prompt = f"""You are an expert legal document analyst performing entity resolution.

I have extracted multiple entity mentions from a legal case document.
I need to determine which refer to the SAME real-world entity.

Entity Type: {group.schema_type}

{chr(10).join(candidate_descriptions)}

For EACH candidate respond with:
1. Should it be merged with the others? (YES/NO)
2. Confidence score (0.0 to 1.0)
3. Brief reason (1-2 sentences)

Rules:
- Abbreviations and full names of the same entity → MERGE
- OCR spelling variations of the same entity → MERGE
- Different organizational levels → evaluate carefully
- Roles/positions are PERSONS not organizations — do not merge with orgs
- Genuinely different entities → do NOT merge

Respond in this exact JSON format:
{{
  "candidates": [
    {{
      "candidate_number": 1,
      "vote": "YES",
      "confidence": 0.92,
      "reason": "Clear abbreviation of the same statutory body"
    }}
  ],
  "suggested_canonical": "best canonical name for merged entities"
}}"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            result = json.loads(response.choices[0].message.content)

            for item in result.get("candidates", []):
                idx = item["candidate_number"] - 1
                if 0 <= idx < len(group.candidates):
                    group.candidates[idx].llm_vote = item["vote"]
                    group.candidates[idx].llm_merge_confidence = item["confidence"]
                    group.candidates[idx].llm_reason = item["reason"]

            if result.get("suggested_canonical"):
                group.canonical_name = result["suggested_canonical"]

            self._determine_resolution(group)

        except Exception as e:
            logger.error(f"LLM scoring failed for {group.group_id}: {e}")
            group.resolution = Resolution.NEEDS_REVIEW

    def _determine_resolution(self, group: ResolutionGroup):
        """Decide AUTO_MERGE / NEEDS_REVIEW / AUTO_KEEP."""
        confidences = [c.llm_merge_confidence for c in group.candidates]
        votes = [c.llm_vote for c in group.candidates]

        all_yes = all(v == "YES" for v in votes)
        all_no = all(v == "NO" for v in votes)
        all_high = all(c >= AUTO_MERGE_CONFIDENCE for c in confidences)
        all_low = all(c < AUTO_KEEP_CONFIDENCE for c in confidences)

        if all_yes and all_high:
            group.resolution = Resolution.AUTO_MERGE
        elif all_no and all_low:
            group.resolution = Resolution.AUTO_KEEP
        else:
            group.resolution = Resolution.NEEDS_REVIEW

    # ── Step 4: Save state for web UI ─────────────────────────────────────────

    def save_resolution_state(
        self,
        groups: list[ResolutionGroup],
        output_path: str | Path,
    ):
        """Save resolution state to JSON for the review UI."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        auto_merge   = [g for g in groups if g.resolution == Resolution.AUTO_MERGE]
        needs_review = [g for g in groups if g.resolution == Resolution.NEEDS_REVIEW]
        auto_keep    = [g for g in groups if g.resolution == Resolution.AUTO_KEEP]

        state = {
            "created_at": datetime.now().isoformat(),
            "summary": {
                "total_groups":      len(groups),
                "auto_merge_count":  len(auto_merge),
                "needs_review_count": len(needs_review),
                "auto_keep_count":   len(auto_keep),
            },
            "auto_merge":   [self._group_to_dict(g) for g in auto_merge],
            "needs_review": [self._group_to_dict(g) for g in needs_review],
            "auto_keep":    [self._group_to_dict(g) for g in auto_keep],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Resolution state saved → {output_path}\n"
            f"  Auto-merge:   {len(auto_merge)}\n"
            f"  Needs review: {len(needs_review)}\n"
            f"  Auto-keep:    {len(auto_keep)}"
        )

    def _group_to_dict(self, group: ResolutionGroup) -> dict:
        return {
            "group_id":        group.group_id,
            "schema_type":     group.schema_type,
            "resolution":      group.resolution.value,
            "canonical_name":  group.canonical_name,
            "is_complex":      group.is_complex,
            "llm_model_used":  group.llm_model_used,
            "human_decision":  group.human_decision,
            "human_selected":  group.human_selected,
            "disagreement":    group.disagreement,
            "decided_at":      group.decided_at,
            "candidates": [
                {
                    "canonical_name":       c.canonical_name,
                    "text":                 c.text,
                    "schema_type":          c.schema_type,
                    "source_pdf":           c.source_pdf,
                    "source_page":          c.source_page,
                    "source_line":          c.source_line,
                    "context":              c.context,
                    "confidence":           c.confidence,
                    "llm_vote":             c.llm_vote,
                    "llm_merge_confidence": c.llm_merge_confidence,
                    "llm_reason":           c.llm_reason,
                }
                for c in group.candidates
            ],
        }

    # ── Step 5: Apply decisions to Neo4j ──────────────────────────────────────

    def apply_decisions(
        self,
        state_path: str | Path,
        log_path: str | Path,
    ):
        """
        Apply final decisions to Neo4j.
        Called after human review confirms all decisions.
        """
        from pipeline.graph_builder import GraphBuilder

        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        log_lines = [
            f"\n{'='*60}",
            f"Decision log — {datetime.now().isoformat()}",
            f"{'='*60}\n",
        ]

        with GraphBuilder() as builder:
            driver = builder._get_driver()

            # Apply auto-merges first
            for group in state["auto_merge"]:
                self._merge_in_neo4j(driver, group, log_lines)

            # Apply human decisions
            for group in state["needs_review"]:
                decision = group.get("human_decision")
                if decision == "MERGE":
                    self._merge_in_neo4j(driver, group, log_lines)
                    if group.get("disagreement"):
                        log_lines.append(
                            f"[DISAGREEMENT] {group['group_id']}: "
                            f"Human overrode LLM → {group['canonical_name']}"
                        )
                elif decision == "KEEP":
                    log_lines.append(
                        f"[KEPT SEPARATE] {group['group_id']}: "
                        f"{[c['text'] for c in group['candidates']]}"
                    )
                # SKIP or None — no action

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")

        logger.info(f"Decisions applied. Log → {log_path}")

    def _merge_in_neo4j(self, driver, group: dict, log_lines: list):
        """Merge duplicate nodes into canonical node in Neo4j."""
        canonical    = group["canonical_name"]
        schema_type  = group["schema_type"]
        candidates   = group["candidates"]

        with driver.session() as session:
            for candidate in candidates:
                name = candidate["canonical_name"]
                if name == canonical:
                    continue

                # Redirect outgoing relationships
                session.run(f"""
                    MATCH (old:{schema_type} {{canonicalName: $old_name}})
                    MATCH (canon:{schema_type} {{canonicalName: $canonical}})
                    WITH old, canon
                    MATCH (old)-[r]->(target)
                    WHERE target <> canon
                    MERGE (canon)-[:RELATED_TO]->(target)
                    DELETE r
                """, old_name=name, canonical=canonical)

                # Redirect incoming relationships
                session.run(f"""
                    MATCH (old:{schema_type} {{canonicalName: $old_name}})
                    MATCH (canon:{schema_type} {{canonicalName: $canonical}})
                    WITH old, canon
                    MATCH (source)-[r]->(old)
                    WHERE source <> canon
                    MERGE (source)-[:RELATED_TO]->(canon)
                    DELETE r
                """, old_name=name, canonical=canonical)

                # Delete duplicate node
                session.run(f"""
                    MATCH (old:{schema_type} {{canonicalName: $old_name}})
                    DETACH DELETE old
                """, old_name=name)

                log_lines.append(
                    f"[MERGED] \"{name}\" → \"{canonical}\" [{schema_type}]"
                )


# ── Entry point ────────────────────────────────────────────────────────────────

def run_resolver(extraction_json_path: str | Path):
    """CLI entry point for manual resolver runs."""
    logging.basicConfig(level=logging.INFO)
    from config import settings
    from pathlib import Path as P

    extraction_path = P(extraction_json_path)
    case_id = extraction_path.parent.name
    output_path = settings.case_resolution_state(case_id)

    resolver = EntityResolver()
    resolver.resolve(extraction_path, output_path)

    logger.info(f"\nDone. Open the review UI at http://localhost:3000/review/{case_id}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline/entity_resolver.py <extraction_json>")
        sys.exit(1)
    run_resolver(sys.argv[1])