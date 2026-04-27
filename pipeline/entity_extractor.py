"""
pipeline/entity_extractor.py
-----------------------------
Stage 2 of the ingestion pipeline.

Responsibilities:
- Run GLiNER over each PageChunk to extract entities
- Map extracted entities to our universal schema types
- Attach source page + confidence score to every entity
- Extract TYPED relationships using LLM (not CO_OCCURS_WITH)

Change from previous version:
    The old _infer_relationships() method created CO_OCCURS_WITH
    for every entity pair on the same page — producing thousands of
    meaningless weak relationships.

    Now replaced with LLM-based typed relationship extraction via
    pipeline/relationship_extractor.py. Every relationship in the
    graph is semantically meaningful. CO_OCCURS_WITH is gone.

    use_llm_relationships=True  → LLM typed extraction (recommended)
    use_llm_relationships=False → skip relationships entirely (fast mode)
"""

from __future__ import annotations

import asyncio
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

UNIVERSAL_LABELS = [
    "Person",
    "Organization",
    "Government Body",
    "Court",
    "Statutory Authority",
    "Financial Institution",
    "Asset",
    "Property",
    "Agreement",
    "Contract",
    "Demand Notice",
    "Legal Proceeding",
    "Writ Petition",
    "Appeal",
    "Court Order",
    "Transaction",
    "Loan",
    "Mortgage",
    "Auction",
    "Payment",
    "Financial Charge",
    "Regulation",
    "Act",
    "Legal Section",
    "Government Resolution",
    "Policy",
    "Date",
    "Amount",
    "Case Number",
    "Document Reference",
    "Location",
    "Address",
]

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
    text: str
    canonical_name: str
    schema_type: str
    gliner_label: str
    confidence: float
    source_pdf: str
    source_page: int
    context: str = ""
    source_doc_id: str = ""
    source_filename: str = ""


@dataclass
class ExtractedRelationship:
    from_entity: str
    to_entity: str
    relation_type: str
    source_pdf: str
    source_page: int
    confidence: float
    source_doc_id: str = ""
    source_filename: str = ""


@dataclass
class ExtractionResult:
    source_pdf: str
    total_pages: int
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)

    @property
    def unique_entities(self) -> list[ExtractedEntity]:
        seen = {}
        for e in self.entities:
            key = (e.canonical_name, e.schema_type)
            if key not in seen or e.confidence > seen[key].confidence:
                seen[key] = e
        return list(seen.values())


# ── Core Extractor ─────────────────────────────────────────────────────────────

class EntityExtractor:
    """
    Runs GLiNER over PageChunks to extract structured entities,
    then uses LLM to extract typed relationships between them.

    Usage:
        extractor = EntityExtractor()
        result = extractor.extract_from_tree(doc_tree)

        # Fast mode — entities only, no relationships
        extractor = EntityExtractor(use_llm_relationships=False)
    """

    def __init__(
        self,
        model_name: str = "urchade/gliner_mediumv2.1",
        labels: list[str] = None,
        threshold: float = 0.4,
        batch_size: int = 8,
        use_llm_relationships: bool = True,
    ):
        self.model_name = model_name
        self.labels = labels or UNIVERSAL_LABELS
        self.threshold = threshold
        self.batch_size = batch_size
        self.use_llm_relationships = use_llm_relationships
        self._model = None
        self._rel_extractor = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading GLiNER model: {self.model_name}")
            try:
                from gliner import GLiNER
                self._model = GLiNER.from_pretrained(self.model_name)
                logger.info("GLiNER model loaded successfully.")
            except ImportError:
                raise ImportError("GLiNER not installed. Run: poetry add gliner")
        return self._model

    def _get_rel_extractor(self):
        if self._rel_extractor is None:
            from pipeline.relationship_extractor import RelationshipExtractor
            self._rel_extractor = RelationshipExtractor()
        return self._rel_extractor

    def extract_from_tree(self, doc_tree: DocumentTree) -> ExtractionResult:
        """Extract entities and typed relationships from all pages."""
        model = self._load_model()
        result = ExtractionResult(
            source_pdf=doc_tree.source_pdf,
            total_pages=doc_tree.total_pages,
        )

        logger.info(
            f"Extracting entities from {doc_tree.source_pdf} "
            f"({doc_tree.total_pages} pages)"
        )

        pages = doc_tree.pages
        for i in range(0, len(pages), self.batch_size):
            batch = pages[i: i + self.batch_size]
            for chunk in batch:
                if not chunk.text.strip():
                    continue

                # Step 1 — GLiNER entity extraction
                entities = self._extract_page_entities(model, chunk)
                result.entities.extend(entities)

                # Step 2 — LLM typed relationship extraction (concurrent)
                if self.use_llm_relationships and len(entities) >= 2:
                    from pipeline.relationship_extractor import (
                        extract_relationships_for_page_async,
                    )
                    relationships = asyncio.run(
                        extract_relationships_for_page_async(
                            entities=entities,
                            page_text=chunk.text,
                            source_pdf=chunk.source_pdf,
                            source_page=chunk.page_number,
                            extractor=self._get_rel_extractor(),
                        )
                    )
                    result.relationships.extend(relationships)

            logger.debug(
                f"Processed pages {i+1}–{min(i+self.batch_size, len(pages))}"
                f" of {len(pages)}"
            )

        logger.info(
            f"Extraction complete: "
            f"{len(result.entities)} raw entities, "
            f"{len(result.unique_entities)} unique, "
            f"{len(result.relationships)} typed relationships"
        )
        return result

    def _extract_page_entities(
        self, model, chunk: PageChunk
    ) -> list[ExtractedEntity]:
        """Run GLiNER on a single page chunk."""
        entities = []
        try:
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
        """Normalize entity text for deduplication."""
        name = text.lower().strip()

        if schema_type == "Organization":
            for suffix in [" pvt. ltd.", " pvt ltd", " ltd.", " ltd",
                           " llp", " inc.", " corp.", " limited"]:
                name = name.replace(suffix, "")

        name = name.replace(",", "").replace(".", "").replace("'", "")
        name = "_".join(name.split())
        return name


# ── Convenience Function ───────────────────────────────────────────────────────

def extract_entities(
    doc_tree: DocumentTree,
    model_name: str = "urchade/gliner_mediumv2.1",
    threshold: float = 0.4,
    use_llm_relationships: bool = True,
) -> ExtractionResult:
    extractor = EntityExtractor(
        model_name=model_name,
        threshold=threshold,
        use_llm_relationships=use_llm_relationships,
    )
    return extractor.extract_from_tree(doc_tree)


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

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

    extractor = EntityExtractor(threshold=0.3, use_llm_relationships=True)
    result = extractor.extract_from_tree(fake_tree)

    print(f"\n{'='*60}")
    print(f"Unique entities: {len(result.unique_entities)}")
    print(f"Typed relationships: {len(result.relationships)}")
    print(f"{'='*60}")
    for e in sorted(result.unique_entities, key=lambda x: x.schema_type):
        print(f"  [{e.schema_type}] {e.text} ({e.confidence:.2f})")
    print()
    for r in result.relationships:
        print(f"  ({r.from_entity}) --[{r.relation_type}]--> ({r.to_entity}) [{r.confidence:.2f}]")