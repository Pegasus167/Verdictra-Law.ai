"""
pipeline/entity_management.py
-------------------------------
Confirmed Entity Registry and Incremental Resolver.

This module sits between the extractor and the review page.
It makes adding new documents to an existing case frictionless:

    First document ingested:
        → All entities go through full resolution
        → Lawyer reviews ALL entity groups on the review page
        → Confirmed decisions are saved to entity_registry.json

    Subsequent documents ingested:
        → New entities compared against confirmed registry
        → Already-confirmed entities AUTO-MERGED — no review needed
        → Only genuinely new entities go to review page

Example:
    Case has MIDC confirmed as "Maharashtra Industrial Development Corporation"
    New transcript uploaded mentions "the Authority" and "MIDC" and "new_party_xyz"
    
    entity_management.py:
        "the Authority" → fuzzy match → MIDC (0.92 confidence) → AUTO_MERGE
        "MIDC"          → exact match → MIDC → AUTO_MERGE
        "new_party_xyz" → no match → NEEDS_REVIEW

    Review page shows: only new_party_xyz
    Not: MIDC, the Authority (already confirmed)

Registry stored at: cases/{case_id}/entity_registry.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Confidence threshold for auto-matching against confirmed entities
AUTO_MATCH_THRESHOLD = 0.85   # Above this → auto-merge with confirmed entity
REVIEW_THRESHOLD     = 0.60   # Below this → treat as new entity, send to review


class EntityRegistry:
    """
    Manages the confirmed entity registry for a case.

    The registry is built from human review decisions and grows with each
    document added to the case. It is the institutional memory of the case
    — every entity the lawyer has verified and named.
    """

    def __init__(self, case_path: Path):
        self.case_path     = case_path
        self.registry_path = case_path / "entity_registry.json"

    # ── Registry I/O ──────────────────────────────────────────────────────────

    def load(self) -> dict:
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"confirmed_entities": {}, "created_at": datetime.now().isoformat()}

    def save(self, registry: dict):
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    # ── Build registry from review decisions ──────────────────────────────────

    def build_from_resolution_state(
        self,
        resolution_state_path: Path,
        confirmed_by: str = "system",
    ):
        """
        Build or update the registry from completed review decisions.

        Called after the lawyer clicks "Confirm All" on the review page.
        Takes the resolution_state.json and extracts all confirmed entities
        into the registry.
        """
        with open(resolution_state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        registry = self.load()
        confirmed = registry.get("confirmed_entities", {})

        # Process auto-merge groups
        for group in state.get("auto_merge", []):
            canonical = group.get("canonical_name", "")
            if not canonical:
                continue
            key = canonical.lower().replace(" ", "_")
            aliases = [c["canonical_name"] for c in group.get("candidates", [])]
            confirmed[key] = {
                "canonical_name": canonical,
                "schema_type":    group.get("schema_type", "Entity"),
                "aliases":        list(set(aliases)),
                "confirmed_at":   datetime.now().isoformat(),
                "confirmed_by":   confirmed_by,
                "source_docs":    list(set(
                    c.get("source_pdf", "") for c in group.get("candidates", [])
                )),
                "resolution":     "AUTO_MERGE",
            }

        # Process human-reviewed groups
        for group in state.get("needs_review", []):
            human_decision = group.get("human_decision")
            if human_decision not in ("MERGE", "KEEP"):
                continue  # Skipped groups don't get confirmed

            canonical = group.get("canonical_name", "")
            if not canonical:
                # Use first candidate name if no canonical set
                candidates = group.get("candidates", [])
                canonical = candidates[0]["canonical_name"] if candidates else ""

            if not canonical:
                continue

            key = canonical.lower().replace(" ", "_")
            aliases = [c["canonical_name"] for c in group.get("candidates", [])]
            confirmed[key] = {
                "canonical_name": canonical,
                "schema_type":    group.get("schema_type", "Entity"),
                "aliases":        list(set(aliases)),
                "confirmed_at":   group.get("decided_at", datetime.now().isoformat()),
                "confirmed_by":   confirmed_by,
                "source_docs":    list(set(
                    c.get("source_pdf", "") for c in group.get("candidates", [])
                )),
                "resolution":     human_decision,
            }

        registry["confirmed_entities"] = confirmed
        registry["last_updated"]       = datetime.now().isoformat()
        self.save(registry)

        logger.info(
            f"Entity registry built: {len(confirmed)} confirmed entities "
            f"for case at {self.case_path.name}"
        )
        return confirmed

    # ── Incremental resolution ─────────────────────────────────────────────────

    def classify_new_entities(
        self,
        new_entities: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """
        Classify newly extracted entities against the confirmed registry.

        Returns:
            auto_merged: entities matched to confirmed — no review needed
            needs_review: genuinely new entities — send to review page
        """
        registry   = self.load()
        confirmed  = registry.get("confirmed_entities", {})

        if not confirmed:
            # No confirmed entities yet — all go to review
            return [], new_entities

        auto_merged  = []
        needs_review = []

        for entity in new_entities:
            name = entity.get("canonical_name", "").strip()
            if not name:
                continue

            match, score = self._find_best_match(name, confirmed)

            if match and score >= AUTO_MATCH_THRESHOLD:
                # High confidence match — auto-merge with confirmed entity
                entity["matched_to"]        = match["canonical_name"]
                entity["match_confidence"]  = score
                entity["resolution"]        = "AUTO_MERGE_FROM_REGISTRY"
                auto_merged.append(entity)
                logger.debug(
                    f"Auto-merged: '{name}' → '{match['canonical_name']}' "
                    f"(score: {score:.2f})"
                )

            elif match and score >= REVIEW_THRESHOLD:
                # Medium confidence — flag for review with suggestion
                entity["suggested_match"]       = match["canonical_name"]
                entity["suggestion_confidence"] = score
                entity["resolution"]            = "NEEDS_REVIEW"
                needs_review.append(entity)

            else:
                # No match — genuinely new entity
                entity["resolution"] = "NEEDS_REVIEW"
                needs_review.append(entity)

        logger.info(
            f"Incremental resolution: "
            f"{len(auto_merged)} auto-merged, "
            f"{len(needs_review)} need review"
        )

        return auto_merged, needs_review

    # ── Matching logic ─────────────────────────────────────────────────────────

    def _find_best_match(
        self,
        name: str,
        confirmed: dict,
    ) -> tuple[Optional[dict], float]:
        """
        Find the best matching confirmed entity for a given name.

        Checks:
        1. Exact match on canonical_name
        2. Exact match on any alias
        3. Fuzzy string similarity on canonical_name and aliases
        """
        name_lower = name.lower().strip()
        best_match = None
        best_score = 0.0

        for key, entity in confirmed.items():
            canonical_lower = entity["canonical_name"].lower().strip()
            aliases_lower   = [a.lower().strip() for a in entity.get("aliases", [])]

            # Exact match
            if name_lower == canonical_lower or name_lower in aliases_lower:
                return entity, 1.0

            # Fuzzy match on canonical name
            score = SequenceMatcher(None, name_lower, canonical_lower).ratio()
            if score > best_score:
                best_score = score
                best_match = entity

            # Fuzzy match on aliases
            for alias in aliases_lower:
                alias_score = SequenceMatcher(None, name_lower, alias).ratio()
                if alias_score > best_score:
                    best_score = alias_score
                    best_match = entity

        return best_match, best_score

    # ── Registry inspection ────────────────────────────────────────────────────

    def get_confirmed_count(self) -> int:
        return len(self.load().get("confirmed_entities", {}))

    def get_all_confirmed(self) -> dict:
        return self.load().get("confirmed_entities", {})

    def is_confirmed(self, name: str) -> bool:
        """Check if a canonical name or alias is already confirmed."""
        _, score = self._find_best_match(name, self.get_all_confirmed())
        return score >= AUTO_MATCH_THRESHOLD