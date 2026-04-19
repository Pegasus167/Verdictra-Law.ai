"""
pipeline/domains/registry.py
------------------------------
Domain Registry for LAW.ai.

Loads domain-specific entity labels, relationship vocabulary, heading patterns,
and LLM prompt examples. Always merges with universal.json first so universal
entities are always extracted regardless of domain.

Usage:
    registry = DomainRegistry()

    # Get merged config for a specific domain
    config = registry.get("property")

    # Use in entity extraction
    labels = config.gliner_labels        # list[str]
    rel_vocab = config.relationship_vocab  # list[str]
    headings = config.heading_patterns   # list[str]
    examples = config.prompt_examples   # list[dict]

    # Get domain choices for upload UI
    choices = registry.list_domains()   # list[{"id": ..., "name": ..., "description": ...}]

Available domains:
    universal           — always loaded, merged into every domain
    property            — Property & Real Estate
    banking_finance     — Banking & Finance
    corporate           — Corporate & Company Law
    criminal            — Criminal Law
    ip_patent           — IP & Patents
    tax                 — Tax Law
    labour              — Labour & Employment
    constitutional      — Constitutional & Writ
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DOMAINS_DIR = Path(__file__).parent


@dataclass
class DomainConfig:
    """Merged configuration for a domain (universal + domain-specific)."""
    domain_id:          str
    name:               str
    description:        str
    gliner_labels:      list[str]           = field(default_factory=list)
    relationship_vocab: list[str]           = field(default_factory=list)
    heading_patterns:   list[str]           = field(default_factory=list)
    prompt_examples:    list[dict]          = field(default_factory=list)


class DomainRegistry:
    """
    Loads and caches domain configurations.

    Always merges universal.json with the requested domain so universal
    entities (Person, Organization, Court etc.) are always extracted.
    """

    # Maps domain_id → JSON filename
    DOMAIN_FILES: dict[str, str] = {
        "property":         "property.json",
        "banking_finance":  "banking_finance.json",
        "corporate":        "corporate.json",
        "criminal":         "criminal.json",
        "ip_patent":        "ip_patent.json",
        "tax":              "tax.json",
        "labour":           "labour.json",
        "constitutional":   "constitutional.json",
    }

    def __init__(self):
        self._cache: dict[str, DomainConfig] = {}
        self._universal: dict = self._load_json("universal.json")

    def _load_json(self, filename: str) -> dict:
        path = DOMAINS_DIR / filename
        if not path.exists():
            logger.warning(f"Domain file not found: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get(self, domain_id: str) -> DomainConfig:
        """
        Get merged DomainConfig for a domain_id.

        Always merges universal config first, then domain-specific on top.
        Deduplicates all lists — no duplicate GLiNER labels.

        Args:
            domain_id: One of the DOMAIN_FILES keys, or "universal" for base only.

        Returns:
            DomainConfig with merged labels, vocab, patterns, examples.
        """
        if domain_id in self._cache:
            return self._cache[domain_id]

        universal = self._universal

        if domain_id == "universal" or domain_id not in self.DOMAIN_FILES:
            if domain_id not in ("universal",):
                logger.warning(
                    f"Unknown domain '{domain_id}' — using universal config only. "
                    f"Available: {list(self.DOMAIN_FILES.keys())}"
                )
            config = self._build_config("universal", universal, {})
        else:
            domain_data = self._load_json(self.DOMAIN_FILES[domain_id])
            config = self._build_config(domain_id, universal, domain_data)

        self._cache[domain_id] = config
        logger.info(
            f"Domain '{domain_id}' loaded: "
            f"{len(config.gliner_labels)} labels, "
            f"{len(config.relationship_vocab)} rel types, "
            f"{len(config.heading_patterns)} heading patterns"
        )
        return config

    def _build_config(
        self,
        domain_id: str,
        universal: dict,
        domain: dict,
    ) -> DomainConfig:
        """Merge universal + domain, deduplicate lists, return DomainConfig."""

        def merge_unique(a: list, b: list) -> list:
            """Merge two lists preserving order and removing duplicates."""
            seen = set()
            result = []
            for item in (a or []) + (b or []):
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result

        return DomainConfig(
            domain_id=domain_id,
            name=domain.get("name", universal.get("name", domain_id)),
            description=domain.get("description", universal.get("description", "")),
            gliner_labels=merge_unique(
                universal.get("gliner_labels", []),
                domain.get("gliner_labels", []),
            ),
            relationship_vocab=merge_unique(
                universal.get("relationship_vocab", []),
                domain.get("relationship_vocab", []),
            ),
            heading_patterns=merge_unique(
                universal.get("heading_patterns", []),
                domain.get("heading_patterns", []),
            ),
            prompt_examples=(
                universal.get("prompt_examples", []) +
                domain.get("prompt_examples", [])
            ),
        )

    def list_domains(self) -> list[dict]:
        """
        Return list of available domains for the upload UI dropdown.

        Returns:
            List of dicts with id, name, description.
        """
        domains = []
        for domain_id, filename in self.DOMAIN_FILES.items():
            data = self._load_json(filename)
            domains.append({
                "id":          domain_id,
                "name":        data.get("name", domain_id),
                "description": data.get("description", ""),
            })
        return sorted(domains, key=lambda d: d["name"])

    def get_for_case(self, case_metadata: dict) -> DomainConfig:
        """
        Get DomainConfig from case metadata dict.
        Convenience method — reads 'domain' field from metadata.json.

        Args:
            case_metadata: Dict loaded from cases/{case_id}/metadata.json

        Returns:
            DomainConfig for the case's domain, or universal if not set.
        """
        domain_id = case_metadata.get("domain", "universal")
        return self.get(domain_id)


# ── Module-level singleton ─────────────────────────────────────────────────────
# Import this directly in pipeline stages

_registry: DomainRegistry | None = None


def get_registry() -> DomainRegistry:
    """Get or create the module-level registry singleton."""
    global _registry
    if _registry is None:
        _registry = DomainRegistry()
    return _registry


def get_domain_config(domain_id: str) -> DomainConfig:
    """Convenience function — get config for a domain_id."""
    return get_registry().get(domain_id)


def get_domain_config_for_case(case_metadata: dict) -> DomainConfig:
    """Convenience function — get config from case metadata."""
    return get_registry().get_for_case(case_metadata)


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    registry = DomainRegistry()

    print("\nAvailable domains:")
    for d in registry.list_domains():
        print(f"  {d['id']:20s} — {d['name']}")

    print("\nProperty domain config:")
    config = registry.get("property")
    print(f"  GLiNER labels ({len(config.gliner_labels)}): {config.gliner_labels[:5]}...")
    print(f"  Rel vocab ({len(config.relationship_vocab)}): {config.relationship_vocab[:5]}...")
    print(f"  Heading patterns ({len(config.heading_patterns)}): {config.heading_patterns[:5]}...")

    print("\nConstitutional domain config (CELIR case):")
    config = registry.get("constitutional")
    print(f"  GLiNER labels ({len(config.gliner_labels)}): {config.gliner_labels[:5]}...")
    print(f"  Rel vocab ({len(config.relationship_vocab)}): {config.relationship_vocab[:5]}...")
