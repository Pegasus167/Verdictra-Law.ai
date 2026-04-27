"""
ingestion.py — Updated pipeline with domain-aware entity extraction
---------------------------------------------------------------------
Stages:
    1. PDF → Markdown (pdf_to_markdown.py)
    2. Markdown → Tree (tree_builder.py)
    3. GLiNER entity extraction using domain-specific labels
    4. Neo4j graph construction (case_id partitioned)
    5. Save extraction.json + pages.json

Domain config is loaded from metadata.json at the start of each case.
Universal labels are always included. Domain-specific labels are merged on top.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import logging
from datetime import datetime
from pathlib import Path

from config import settings
from pipeline.tree_builder import build_tree_from_markdown, save_tree, TreeNode
from pipeline.pdf_extractor import DocumentTree, PageChunk
from pipeline.entity_extractor import EntityExtractor, ExtractedEntity, ExtractedRelationship
from pipeline.graph_builder import GraphBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

PAGE_BATCH_SIZE     = 20
MAX_GLINER_TOKENS   = 300
CHUNK_OVERLAP_WORDS = 30


def entity_to_dict(e: ExtractedEntity) -> dict:
    return {
        "canonical_name": e.canonical_name,
        "text":           e.text,
        "schema_type":    e.schema_type,
        "gliner_label":   e.gliner_label,
        "confidence":     e.confidence,
        "source_pdf":     e.source_pdf,
        "source_page":    e.source_page,
        "context":        e.context,
        "source_doc_id":  getattr(e, "source_doc_id", ""),
        "source_filename": getattr(e, "source_filename", e.source_pdf),
    }


def relationship_to_dict(r: ExtractedRelationship) -> dict:
    return {
        "from_entity":   r.from_entity,
        "to_entity":     r.to_entity,
        "relation_type": r.relation_type,
        "source_pdf":    r.source_pdf,
        "source_page":   r.source_page,
        "confidence":    r.confidence,
        "source_doc_id":  getattr(r, "source_doc_id", ""),
        "source_filename": getattr(r, "source_filename", r.source_pdf),
    }


def _split_into_chunks(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += max_words - overlap
    return chunks


def tree_to_page_chunks(tree: TreeNode, source_pdf: str) -> list[PageChunk]:
    chunks = []

    def collect(node: TreeNode):
        if not node.content or len(node.content.strip()) <= 10:
            for child in node.nodes:
                collect(child)
            return
        full_text  = f"{node.title}\n\n{node.content}".strip()
        sub_chunks = _split_into_chunks(full_text, MAX_GLINER_TOKENS, CHUNK_OVERLAP_WORDS)
        for sub in sub_chunks:
            chunks.append(PageChunk(
                source_pdf=source_pdf,
                page_number=node.page_num or 0,
                text=sub,
                word_count=len(sub.split()),
                has_tables=False,
                extraction_method="tree_node",
            ))
        for child in node.nodes:
            collect(child)

    collect(tree)
    return chunks


def tree_to_pages_json(tree: TreeNode, source_pdf: str) -> list[dict]:
    page_texts: dict[int, list[str]] = {}

    def collect(node: TreeNode):
        if node.content and len(node.content.strip()) > 10:
            page = node.page_num or 0
            text = f"{node.title}\n\n{node.content}".strip()
            page_texts.setdefault(page, []).append(text)
        for child in node.nodes:
            collect(child)

    collect(tree)
    pages = []
    for page_num in sorted(page_texts.keys()):
        full_text = "\n\n".join(page_texts[page_num])
        pages.append({
            "page_number":       page_num,
            "text":              full_text,
            "word_count":        len(full_text.split()),
            "source_pdf":        source_pdf,
            "extraction_method": "tree_node",
        })
    return pages


def save_pages_json(pages: list[dict], case_id: str):
    output_path = settings.case_path(case_id) / "pages.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    logger.info(f"Pages saved → {output_path} ({len(pages)} pages)")


def save_extraction_json(all_entities, all_relationships, source_pdf, case_id):
    output_path = settings.case_extraction(case_id)
    seen = {}
    for e in all_entities:
        key = (e.canonical_name, e.schema_type)
        if key not in seen or e.confidence > seen[key].confidence:
            seen[key] = e
    unique_entities = list(seen.values())

    payload = {
        "source_pdf":          source_pdf,
        "extracted_at":        datetime.now().isoformat(),
        "total_entities":      len(unique_entities),
        "total_relationships": len(all_relationships),
        "entities":            [entity_to_dict(e) for e in unique_entities],
        "relationships":       [relationship_to_dict(r) for r in all_relationships],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Extraction JSON saved → {output_path}")
    return output_path


def _load_domain_labels(case_id: str) -> list[str] | None:
    """
    Load GLiNER labels for this case from metadata.json domain field.
    Returns merged (universal + domain) label list.
    Falls back to None if registry unavailable — EntityExtractor uses its defaults.
    """
    try:
        from pipeline.domains.registry import DomainRegistry
        meta_path = settings.case_metadata(case_id)
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            domain_id = meta.get("domain", "constitutional")
        else:
            domain_id = "constitutional"

        registry = DomainRegistry()
        config   = registry.get(domain_id)
        logger.info(
            f"Domain '{domain_id}' — "
            f"{len(config.gliner_labels)} GLiNER labels loaded"
        )
        return config.gliner_labels
    except Exception as e:
        logger.warning(f"Domain label load failed ({e}) — EntityExtractor uses defaults")
        return None


def run_pipeline(path: str | Path, case_id: str = None, use_ocr: bool = True, doc_id: str = "", doc_filename: str = "",):
    path = Path(path)
    settings.ensure_dirs()

    pdfs = list(path.glob("*.pdf")) if path.is_dir() else [path]
    if not pdfs:
        logger.warning("No PDFs found.")
        return

    with GraphBuilder() as builder:
        builder.setup_constraints()

        for pdf_path in pdfs:
            logger.info(f"\n{'='*60}\nProcessing: {pdf_path.name}\n{'='*60}")
            _case_id = case_id or pdf_path.stem
            settings.ensure_case_dirs(_case_id)

            def update_stage(stage: int):
                try:
                    meta_path = settings.case_metadata(_case_id)
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["current_stage"] = stage
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                except Exception:
                    pass

            try:
                # Load domain-specific GLiNER labels for this case
                domain_labels = _load_domain_labels(_case_id)

                # Build extractor with domain labels if available
                if domain_labels:
                    extractor = EntityExtractor(
                        model_name=settings.gliner_model,
                        threshold=0.5,
                        labels=domain_labels,
                    )
                else:
                    extractor = EntityExtractor(
                        model_name=settings.gliner_model,
                        threshold=0.5,
                    )
                    
                # Stage 1: File → Markdown (any supported format)
                logger.info(f"Stage 1: Extracting {pdf_path.name}...")
                from pipeline.extractors import extract_file, is_supported

                if not is_supported(pdf_path):
                    raise ValueError(
                        f"Unsupported file type: {pdf_path.suffix}. "
                        f"Upload PDF, DOCX, EML, TXT, or image files."
                    )

                extraction_output = extract_file(pdf_path, use_ocr=use_ocr)
                md_text = extraction_output.markdown_text

                # Save markdown for debugging/reference
                md_output = settings.case_path(_case_id) / f"{doc_filename or pdf_path.stem}.md"
                with open(md_output, "w", encoding="utf-8") as f:
                    f.write(md_text)

                update_stage(2)

                # Stage 2: Markdown → Tree
                logger.info("Stage 2: Markdown → Tree...")
                tree = build_tree_from_markdown(md_text)
                save_tree(tree, settings.case_path(_case_id) / "tree.json")
                logger.info(f"  → {tree.node_count()} nodes")
                update_stage(3)

                # Stage 3: Save pages.json
                logger.info("Stage 3: Saving pages.json...")
                pages = tree_to_pages_json(tree, pdf_path.name)
                save_pages_json(pages, _case_id)
                update_stage(4)

                # Stage 4: GLiNER + relationship extraction on tree nodes
                logger.info("Stage 4: Entity extraction on tree nodes...")
                chunks            = tree_to_page_chunks(tree, pdf_path.name)
                all_entities      = []
                all_relationships = []
                # Instantiate validator once per case
                from pipeline.validator import EntityValidator
                validator = EntityValidator(min_confidence=0.5, min_name_length=3)

                for i in range(0, len(chunks), PAGE_BATCH_SIZE):
                    batch      = chunks[i: i + PAGE_BATCH_SIZE]
                    batch_tree = DocumentTree(
                        source_pdf=pdf_path.name,
                        total_pages=len(batch),
                        pages=batch,
                        sections=[],
                    )
                    batch_result = extractor.extract_from_tree(batch_tree)

                    # ── Stamp doc_id and filename on every entity/relationship ──
                    _doc_id       = doc_id or f"doc_001"
                    _doc_filename = doc_filename or pdf_path.name
                    for e in batch_result.entities:
                        e.source_doc_id   = _doc_id
                        e.source_filename = _doc_filename
                    for r in batch_result.relationships:
                        r.source_doc_id   = _doc_id
                        r.source_filename = _doc_filename
                    
                    # ── Validate entities before writing to Neo4j ──────────────
                    valid_entities, rejected = validator.validate_batch(
                        batch_result.entities
                    )
                    if rejected:
                        rejection_stats = validator.get_stats(rejected)
                        logger.info(f"  Validation rejected {len(rejected)} entities: {rejection_stats}")

                    # Replace entities in batch_result with validated ones only
                    batch_result.entities = valid_entities
                    # Also filter unique_entities
                    batch_result._unique_entities = None  # force recompute

                    builder.build_from_extraction(batch_result, case_id=_case_id)
                    all_entities.extend(batch_result.entities)
                    all_relationships.extend(batch_result.relationships)
                    logger.info(
                        f"  ✓ Nodes {i+1}–{min(i+PAGE_BATCH_SIZE, len(chunks))}: "
                        f"{len(batch_result.unique_entities)} entities"
                    )

                # Stage 5: Save extraction.json
                logger.info("Stage 5: Saving extraction.json...")
                save_extraction_json(
                    all_entities, all_relationships, pdf_path.name, _case_id
                )
                update_stage(5)

                # Update metadata
                meta_path = settings.case_metadata(_case_id)
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["pages"]    = tree.node_count()
                    meta["status"]   = "review"
                    meta["has_tree"] = True
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)

                stats = builder.get_stats(case_id=_case_id)
                logger.info(
                    f"Graph [{_case_id}]: "
                    f"{stats['total_nodes']} nodes, "
                    f"{stats['total_relationships']} relationships"
                )

            except Exception as e:
                logger.error(f"Failed {pdf_path.name}: {e}")
                import traceback; traceback.print_exc()

    logger.info("\nIngestion complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <pdf_path> [case_id]")
        sys.exit(1)
    run_pipeline(sys.argv[1], case_id=sys.argv[2] if len(sys.argv) > 2 else None)