"""
pipeline/entity_extractor.py
-----------------------------
Stage 2 of the ingestion pipeline.

Responsibilities:
- Run GLiNER over each PageChunk to extract entities
- Map extracted entities to our universal schema types
- Attach source page + confidence score to every entity
- Detect relationships between co-occurring entities on same page
- Output structured ExtractedEntity and ExtractedRelationship objects

Why GLiNER:
    Zero-shot — no retraining needed per case type.
    You just pass label strings and it finds matching spans.
    Works for any legal/financial document without case-specific tuning.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.pdf_extractor import PageChunk, DocumentTree

logger = logging.getLogger(__name__)


# ── Universal Entity Labels ────────────────────────────────────────────────────
# These are the labels passed to GLiNER at inference time.
# Add new labels here as needed — no retraining required.

UNIVERSAL_LABELS = [
    # Core parties
    "Person",
    "Organization",
    "Government Body",
    "Court",
    "Statutory Authority",
    "Financial Institution",

    # Case assets + documents
    "Asset",
    "Property",
    "Agreement",
    "Contract",
    "Demand Notice",
    "Legal Proceeding",
    "Writ Petition",
    "Appeal",
    "Court Order",

    # Financial
    "Transaction",
    "Loan",
    "Mortgage",
    "Auction",
    "Payment",
    "Financial Charge",

    # Legal + Regulatory
    "Regulation",
    "Act",
    "Legal Section",
    "Government Resolution",
    "Policy",

    # Identifiers
    "Date",
    "Amount",
    "Case Number",
    "Document Reference",
    "Location",
    "Address",
]

# Map GLiNER raw labels → our canonical schema types
# This handles label aliases and normalizes to the universal schema
LABEL_TO_SCHEMA_TYPE = {
    "Person": "Person",
    "Organization": "Organization",
    "Government Body": "Government",
    "Court": "Court",
    "Statutory Authority": "StatutoryBody",
    "Financial Institution": "FinancialInstitution",
    "Asset": "Asset",
    "Property": "Asset",
    "Agreement": "Agreement",
    "Contract": "Agreement",
    "Demand Notice": "Demand",
    "Legal Proceeding": "LegalProceeding",
    "Writ Petition": "LegalProceeding",
    "Appeal": "LegalProceeding",
    "Court Order": "LegalProceeding",
    "Transaction": "Transaction",
    "Loan": "Transaction",
    "Mortgage": "Transaction",
    "Auction": "Event",
    "Payment": "Transaction",
    "Financial Charge": "Demand",
    "Regulation": "Regulation",
    "Act": "Regulation",
    "Legal Section": "Regulation",
    "Government Resolution": "Regulation",
    "Policy": "Regulation",
    "Date": "Date",
    "Amount": "Amount",
    "Case Number": "Identifier",
    "Document Reference": "Identifier",
    "Location": "Location",
    "Address": "Location",
}


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class ExtractedEntity:
    """
    A single entity extracted from a page chunk.
    This becomes a Neo4j node.
    """
    text: str                  # Raw extracted text (e.g. "CELIR LLP")
    canonical_name: str        # Normalized version for deduplication
    schema_type: str           # Universal schema type (e.g. "Organization")
    gliner_label: str          # Raw GLiNER label (e.g. "Organization")
    confidence: float          # GLiNER confidence score 0.0–1.0
    source_pdf: str            # Which document this came from
    source_page: int           # Exact page number
    context: str               # Surrounding text snippet (for disambiguation)


@dataclass
class ExtractedRelationship:
    """
    A co-occurrence based relationship between two entities on the same page.
    This becomes a Neo4j edge (refined later by LLM if needed).
    """
    from_entity: str           # canonical_name of source entity
    to_entity: str             # canonical_name of target entity
    relation_type: str         # e.g. "CO_OCCURS_WITH" or inferred type
    source_pdf: str
    source_page: int
    confidence: float          # Lower than entity confidence — inferred


@dataclass
class ExtractionResult:
    """Full extraction output for one document."""
    source_pdf: str
    total_pages: int
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)

    @property
    def unique_entities(self) -> list[ExtractedEntity]:
        """Deduplicated entities by canonical_name + schema_type."""
        seen = {}
        for e in self.entities:
            key = (e.canonical_name, e.schema_type)
            # Keep highest confidence occurrence
            if key not in seen or e.confidence > seen[key].confidence:
                seen[key] = e
        return list(seen.values())


# ── Core Extractor ─────────────────────────────────────────────────────────────

class EntityExtractor:
    """
    Runs GLiNER over PageChunks to extract structured entities.

    Usage:
        extractor = EntityExtractor()
        result = extractor.extract_from_tree(doc_tree)
        for entity in result.unique_entities:
            print(f"{entity.schema_type}: {entity.canonical_name} (p{entity.source_page})")
    """

    def __init__(
        self,
        model_name: str = "urchade/gliner_mediumv2.1",
        labels: list[str] = None,
        threshold: float = 0.4,       # Minimum confidence to keep entity
        batch_size: int = 8,           # Pages processed per GLiNER batch
    ):
        self.model_name = model_name
        self.labels = labels or UNIVERSAL_LABELS
        self.threshold = threshold
        self.batch_size = batch_size
        self._model = None             # Lazy loaded

    def _load_model(self):
        """Load GLiNER model (lazy — only on first use)."""
        if self._model is None:
            logger.info(f"Loading GLiNER model: {self.model_name}")
            try:
                from gliner import GLiNER
                self._model = GLiNER.from_pretrained(self.model_name)
                logger.info("GLiNER model loaded successfully.")
            except ImportError:
                raise ImportError(
                    "GLiNER not installed. Run: poetry add gliner"
                )
        return self._model

    def extract_from_tree(self, doc_tree: DocumentTree) -> ExtractionResult:
        """
        Extract entities from all pages of a document.

        Args:
            doc_tree: DocumentTree from PDFExtractor

        Returns:
            ExtractionResult with all entities + relationships
        """
        model = self._load_model()
        result = ExtractionResult(
            source_pdf=doc_tree.source_pdf,
            total_pages=doc_tree.total_pages,
        )

        logger.info(
            f"Extracting entities from {doc_tree.source_pdf} "
            f"({doc_tree.total_pages} pages)"
        )

        # Process pages in batches
        pages = doc_tree.pages
        for i in range(0, len(pages), self.batch_size):
            batch = pages[i : i + self.batch_size]
            for chunk in batch:
                if not chunk.text.strip():
                    continue

                entities = self._extract_page_entities(model, chunk)
                result.entities.extend(entities)

                # Detect relationships from co-occurrences on same page
                relationships = self._infer_relationships(entities, chunk)
                result.relationships.extend(relationships)

            logger.debug(
                f"Processed pages {i+1}–{min(i+self.batch_size, len(pages))}"
                f" of {len(pages)}"
            )

        logger.info(
            f"Extraction complete: "
            f"{len(result.entities)} raw entities, "
            f"{len(result.unique_entities)} unique, "
            f"{len(result.relationships)} relationships"
        )
        return result

    def _extract_page_entities(
        self, model, chunk: PageChunk
    ) -> list[ExtractedEntity]:
        """Run GLiNER on a single page and return structured entities."""
        entities = []

        try:
            # GLiNER expects: model.predict_entities(text, labels, threshold)
            raw_entities = model.predict_entities(
                chunk.text,
                self.labels,
                threshold=self.threshold,
            )

            for raw in raw_entities:
                text = raw["text"].strip()
                label = raw["label"]
                score = raw["score"]

                if not text or len(text) < 2:
                    continue

                schema_type = LABEL_TO_SCHEMA_TYPE.get(label, "Entity")
                canonical = self._canonicalize(text, schema_type)

                # Get surrounding context (50 chars either side)
                start = max(0, chunk.text.find(text) - 50)
                end = min(len(chunk.text), chunk.text.find(text) + len(text) + 50)
                context = chunk.text[start:end].replace("\n", " ")

                entities.append(ExtractedEntity(
                    text=text,
                    canonical_name=canonical,
                    schema_type=schema_type,
                    gliner_label=label,
                    confidence=round(score, 4),
                    source_pdf=chunk.source_pdf,
                    source_page=chunk.page_number,
                    context=context,
                ))

        except Exception as e:
            logger.error(
                f"GLiNER failed on page {chunk.page_number} "
                f"of {chunk.source_pdf}: {e}"
            )

        return entities

    def _canonicalize(self, text: str, schema_type: str) -> str:
        """
        Normalize entity text for deduplication.

        Examples:
            "CELIR LLP" → "celir_llp"
            "Union Bank of India" → "union_bank_of_india"
            "₹14,27,26,992" → "inr_142726992"  (amounts normalized)
        """
        name = text.lower().strip()

        # Remove common legal suffixes for org deduplication
        # (but keep them in the original text)
        if schema_type == "Organization":
            for suffix in [" pvt. ltd.", " pvt ltd", " ltd.", " ltd",
                          " llp", " inc.", " corp.", " limited"]:
                name = name.replace(suffix, "")

        # Normalize punctuation
        name = name.replace(",", "").replace(".", "").replace("'", "")
        name = "_".join(name.split())

        return name

    def _infer_relationships(
        self,
        entities: list[ExtractedEntity],
        chunk: PageChunk,
    ) -> list[ExtractedRelationship]:
        """
        Infer basic relationships from entity co-occurrence on the same page.

        These are weak relationships — labelled CO_OCCURS_WITH.
        The LLM reasoning layer upgrades these to specific typed
        relationships (FILED_BY, TRANSFERRED_TO, etc.) later.
        """
        relationships = []

        # Only infer between meaningful entity types
        meaningful_types = {
            "Person", "Organization", "Court", "StatutoryBody",
            "FinancialInstitution", "Asset", "LegalProceeding", "Demand"
        }

        meaningful = [
            e for e in entities if e.schema_type in meaningful_types
        ]

        # Create co-occurrence pairs (limit to avoid explosion)
        # On a single page, max ~10 meaningful entities = ~45 pairs
        for i, e1 in enumerate(meaningful):
            for e2 in meaningful[i + 1 :]:
                if e1.canonical_name == e2.canonical_name:
                    continue

                relationships.append(ExtractedRelationship(
                    from_entity=e1.canonical_name,
                    to_entity=e2.canonical_name,
                    relation_type="CO_OCCURS_WITH",
                    source_pdf=chunk.source_pdf,
                    source_page=chunk.page_number,
                    confidence=round(
                        min(e1.confidence, e2.confidence) * 0.7, 4
                    ),  # Co-occurrence confidence is lower than entity confidence
                ))

        return relationships


# ── Convenience Function ───────────────────────────────────────────────────────

def extract_entities(
    doc_tree: DocumentTree,
    model_name: str = "urchade/gliner_mediumv2.1",
    threshold: float = 0.4,
) -> ExtractionResult:
    """Convenience wrapper."""
    extractor = EntityExtractor(model_name=model_name, threshold=threshold)
    return extractor.extract_from_tree(doc_tree)


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    # Test on a sample text without needing a real PDF
    from pipeline.pdf_extractor import PageChunk, DocumentTree

    sample_text = """
    CELIR LLP, a Limited Liability Partnership registered under the LLP Act 2008,
    filed a Writ Petition (St. No. 31414 of 2024) in the High Court of Judicature
    at Bombay challenging a demand notice dated 16 January 2024 issued by MIDC
    (Maharashtra Industrial Development Corporation) for Rs. 14,27,26,992.
    The Supreme Court of India in Civil Appeal No. 5542 of 2023 had declared
    CELIR LLP as the rightful leaseholder of Plots D-105, D-110, and D-111
    at TTC Industrial Area, Mahape, Navi Mumbai.
    Union Bank of India conducted the SARFAESI auction on 27 June 2023.
    """

    # Build a fake DocumentTree for testing
    fake_chunk = PageChunk(
        source_pdf="test_case.pdf",
        page_number=1,
        text=sample_text,
        word_count=len(sample_text.split()),
        has_tables=False,
        extraction_method="text",
    )
    fake_tree = DocumentTree(
        source_pdf="test_case.pdf",
        total_pages=1,
        pages=[fake_chunk],
    )

    extractor = EntityExtractor(threshold=0.3)
    result = extractor.extract_from_tree(fake_tree)

    print(f"\n{'='*60}")
    print(f"Unique entities found: {len(result.unique_entities)}")
    print(f"{'='*60}")
    for e in sorted(result.unique_entities, key=lambda x: x.schema_type):
        print(f"  [{e.schema_type}] {e.text} (confidence: {e.confidence})")
    print(f"\nRelationships: {len(result.relationships)}")