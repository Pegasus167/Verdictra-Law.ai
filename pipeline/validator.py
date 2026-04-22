"""
pipeline/validator.py
----------------------
Entity validation layer between GLiNER extraction and Neo4j.

Uses pyshacl to validate extracted entities against shapes.ttl.
Entities that fail validation are rejected before they touch the graph.

This eliminates:
    - Garbage OCR artifacts ("you", "hon", "mavarashcra")
    - Entities with missing required fields
    - Entities with confidence below threshold
    - Single character / purely numeric extractions
    - Common stopwords tagged as entities

Architecture:
    GLiNER extracts entities
        ↓
    EntityValidator.validate_batch() — SHACL validation (free, deterministic)
        ↓  rejects invalid entities with plain-English reasons
    LLM relationship extraction — only runs on CLEAN entities
        ↓  fewer pairs = fewer tokens = lower cost
    Neo4j — only stores validated entities

Usage:
    from pipeline.validator import EntityValidator
    validator = EntityValidator()
    valid_entities, rejected = validator.validate_batch(entities)
    # valid_entities → proceed to graph_builder
    # rejected → logged, never reach Neo4j
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Paths to ontology files
_ROOT = Path(__file__).parent.parent
SHAPES_PATH   = _ROOT / "shapes.ttl"
ONTOLOGY_PATH = _ROOT / "ontology.ttl"

# Noise words — entities whose canonical name is purely boilerplate
_NOISE_WORDS = {
    "the", "and", "or", "of", "in", "at", "to", "for", "by", "with",
    "vs", "v", "re", "ex", "per", "viz", "etc", "ibid", "id",
    "hon", "honble", "ld", "sr", "jr", "dr", "mr", "mrs", "ms",
    "iii", "ii", "iv", "vi", "vii", "viii",
    "i", "a", "an",
}

# Schema type → RDF class mapping
# Maps our Neo4j schema types to the ex: namespace in ontology.ttl
_SCHEMA_TO_CLASS = {
    "Person":              "ex:Person",
    "Organization":        "ex:Organization",
    "Government":          "ex:Government",
    "Court":               "ex:Court",
    "StatutoryBody":       "ex:StatutoryBody",
    "FinancialInstitution":"ex:FinancialInstitution",
    "Asset":               "ex:Asset",
    "Agreement":           "ex:Agreement",
    "Demand":              "ex:Demand",
    "LegalProceeding":     "ex:LegalProceeding",
    "Transaction":         "ex:Transaction",
    "Regulation":          "ex:Regulation",
    "Event":               "ex:Event",
    "Location":            "ex:Location",
    "Identifier":          "ex:Identifier",
    "Entity":              "ex:Organization",  # fallback
}


@dataclass
class ValidationResult:
    """Result of validating a single entity."""
    is_valid: bool
    entity_name: str
    schema_type: str
    reason: str = ""          # why rejected (empty if valid)
    violations: list[str] = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


class EntityValidator:
    """
    Validates extracted entities against shapes.ttl using pyshacl.

    Two-stage validation:
        Stage 1 — Fast pre-checks (no pyshacl, pure Python)
            Catches obvious noise before pyshacl overhead
        Stage 2 — SHACL validation (pyshacl)
            Validates against shapes.ttl constraints

    Usage:
        validator = EntityValidator()
        valid, rejected = validator.validate_batch(entities)
    """

    def __init__(
        self,
        shapes_path: Path = SHAPES_PATH,
        ontology_path: Path = ONTOLOGY_PATH,
        min_confidence: float = 0.5,
        min_name_length: int = 3,
    ):
        self.min_confidence  = min_confidence
        self.min_name_length = min_name_length
        self._shapes_graph   = None
        self._shacl_available = False

        # Load shapes graph once at init
        self._load_shapes(shapes_path, ontology_path)

    def _load_shapes(self, shapes_path: Path, ontology_path: Path):
        """Load SHACL shapes graph from shapes.ttl."""
        try:
            from rdflib import Graph, ConjunctiveGraph

            if not shapes_path.exists():
                logger.warning(
                    f"shapes.ttl not found at {shapes_path} — "
                    "SHACL validation disabled, using pre-checks only"
                )
                return

            # Load shapes
            self._shapes_graph = Graph()
            self._shapes_graph.parse(str(shapes_path), format="turtle")

            # Merge ontology into shapes graph if available
            if ontology_path.exists():
                self._shapes_graph.parse(str(ontology_path), format="turtle")

            self._shacl_available = True
            logger.info(
                f"SHACL validator loaded: {len(self._shapes_graph)} triples "
                f"from shapes.ttl + ontology.ttl"
            )

        except Exception as e:
            logger.warning(
                f"SHACL load failed ({e}) — using pre-checks only. "
                "This is non-fatal; entities will still be filtered."
            )

    def validate_batch(
        self, entities: list
    ) -> tuple[list, list[ValidationResult]]:
        """
        Validate a batch of ExtractedEntity objects.

        Returns:
            (valid_entities, rejected_results)
            valid_entities   — list of entities that passed all checks
            rejected_results — list of ValidationResult for rejected entities
        """
        valid    = []
        rejected = []

        for entity in entities:
            result = self._validate_one(entity)
            if result.is_valid:
                valid.append(entity)
            else:
                rejected.append(result)
                logger.debug(
                    f"Rejected [{entity.schema_type}] '{entity.canonical_name}': "
                    f"{result.reason}"
                )

        if rejected:
            logger.info(
                f"Validation: {len(valid)} accepted, {len(rejected)} rejected "
                f"({len(rejected)/(len(valid)+len(rejected))*100:.0f}% noise rate)"
            )

        return valid, rejected

    def _validate_one(self, entity) -> ValidationResult:
        """Validate a single entity. Returns ValidationResult."""

        name = (entity.canonical_name or "").strip()
        schema_type = entity.schema_type or "Entity"

        # ── Stage 1: Fast pre-checks ───────────────────────────────────────────

        # 1a. Minimum name length
        if len(name) < self.min_name_length:
            return ValidationResult(
                is_valid=False,
                entity_name=name,
                schema_type=schema_type,
                reason=f"Name too short ({len(name)} chars, min {self.min_name_length})",
            )

        # 1b. Purely numeric
        if name.replace(".", "").replace(",", "").replace(" ", "").isdigit():
            return ValidationResult(
                is_valid=False,
                entity_name=name,
                schema_type=schema_type,
                reason="Purely numeric entity",
            )

        # 1c. Noise words
        if name.lower() in _NOISE_WORDS:
            return ValidationResult(
                is_valid=False,
                entity_name=name,
                schema_type=schema_type,
                reason=f"Common noise word: '{name}'",
            )

        # 1d. Confidence threshold
        confidence = getattr(entity, "confidence", 0.0) or 0.0
        if confidence < self.min_confidence:
            return ValidationResult(
                is_valid=False,
                entity_name=name,
                schema_type=schema_type,
                reason=f"Confidence {confidence:.2f} below threshold {self.min_confidence}",
            )

        # 1e. Must have source page
        source_page = getattr(entity, "source_page", None)
        if not source_page:
            return ValidationResult(
                is_valid=False,
                entity_name=name,
                schema_type=schema_type,
                reason="Missing source_page",
            )

        # 1f. Must have source PDF
        source_pdf = getattr(entity, "source_pdf", None)
        if not source_pdf:
            return ValidationResult(
                is_valid=False,
                entity_name=name,
                schema_type=schema_type,
                reason="Missing source_pdf",
            )

        # ── Stage 2: SHACL validation ──────────────────────────────────────────
        if self._shacl_available:
            shacl_result = self._validate_shacl(entity)
            if not shacl_result.is_valid:
                return shacl_result

        return ValidationResult(
            is_valid=True,
            entity_name=name,
            schema_type=schema_type,
            reason="",
        )

    def _validate_shacl(self, entity) -> ValidationResult:
        """
        Validate entity against shapes.ttl using pyshacl.
        Converts entity to minimal RDF graph and validates.
        """
        try:
            import pyshacl
            from rdflib import Graph, Literal, URIRef, Namespace
            from rdflib.namespace import XSD, RDF

            EX = Namespace("http://example.org/ont/case#")

            # Build minimal RDF graph for this entity
            data_graph = Graph()
            data_graph.bind("ex", EX)

            entity_uri = URIRef(
                f"http://example.org/entity/{entity.canonical_name.replace(' ', '_')}"
            )

            # Get RDF class for this schema type
            rdf_class_str = _SCHEMA_TO_CLASS.get(entity.schema_type, "ex:Organization")
            rdf_class     = EX[rdf_class_str.replace("ex:", "")]

            # Add triples
            data_graph.add((entity_uri, RDF.type, rdf_class))
            data_graph.add((entity_uri, EX.canonicalName, Literal(entity.canonical_name, datatype=XSD.string)))
            data_graph.add((entity_uri, EX.sourcePDF, Literal(entity.source_pdf or "", datatype=XSD.string)))
            data_graph.add((entity_uri, EX.sourcePage, Literal(entity.source_page or 0, datatype=XSD.integer)))
            data_graph.add((entity_uri, EX.confidence, Literal(float(entity.confidence or 0), datatype=XSD.decimal)))

            # Run SHACL validation
            conforms, _, results_text = pyshacl.validate(
                data_graph,
                shacl_graph=self._shapes_graph,
                inference="rdfs",
                abort_on_first=True,
                allow_infos=False,
                allow_warnings=False,
            )

            if not conforms:
                # Extract violation message
                violations = [
                    line.strip()
                    for line in results_text.split("\n")
                    if "Message:" in line or "Constraint Violation" in line
                ]
                reason = violations[0] if violations else "SHACL validation failed"
                return ValidationResult(
                    is_valid=False,
                    entity_name=entity.canonical_name,
                    schema_type=entity.schema_type,
                    reason=reason,
                    violations=violations,
                )

            return ValidationResult(
                is_valid=True,
                entity_name=entity.canonical_name,
                schema_type=entity.schema_type,
            )

        except Exception as e:
            # Never let validation crash the pipeline
            # If SHACL validation itself errors, pass the entity through
            logger.debug(f"SHACL validation error for '{entity.canonical_name}': {e}")
            return ValidationResult(
                is_valid=True,
                entity_name=entity.canonical_name,
                schema_type=entity.schema_type,
                reason=f"SHACL error (passed through): {e}",
            )

    def get_stats(self, rejected: list[ValidationResult]) -> dict:
        """Return breakdown of rejection reasons for logging."""
        reasons: dict[str, int] = {}
        for r in rejected:
            key = r.reason.split(":")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
        return reasons