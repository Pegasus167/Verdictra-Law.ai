"""
pipeline/pdf_extractor.py
--------------------------
Stage 1 of the ingestion pipeline.

Responsibilities:
- Extract clean text from PDF (with OCR fallback for scanned pages)
- Track EXACT page number for every text chunk
- Build the document tree structure (PDF → pages → sections → paragraphs)
- Output a list of PageChunk objects ready for GLiNER

Why page tracking matters:
    Every entity extracted later gets a sourcePage property.
    This is what links the Knowledge Graph back to the Document Tree,
    enabling citations like "[Document: Page 52]" in final answers.
"""

import fitz  # PyMuPDF
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

logger = logging.getLogger(__name__)


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class PageChunk:
    """
    A single page's extracted content with full provenance.
    This is the atomic unit that flows into GLiNER for entity extraction.
    """
    source_pdf: str           # Original PDF filename
    page_number: int          # 1-indexed page number
    text: str                 # Clean extracted text
    word_count: int           # Rough quality indicator
    has_tables: bool          # Whether page likely contains tables
    extraction_method: str    # "text" | "ocr" — for debugging


@dataclass
class DocumentTree:
    """
    Hierarchical structure of the entire document.
    Used by the tree retriever for BM25 + hierarchical pruning search.
    """
    source_pdf: str
    total_pages: int
    pages: list[PageChunk] = field(default_factory=list)

    # Detected sections — built after extraction
    sections: list[dict] = field(default_factory=list)


# ── Core Extractor ─────────────────────────────────────────────────────────────

class PDFExtractor:
    """
    Extracts text from PDFs page by page with OCR fallback.

    Usage:
        extractor = PDFExtractor()
        doc_tree = extractor.extract("path/to/case.pdf")
        for chunk in doc_tree.pages:
            print(f"Page {chunk.page_number}: {chunk.text[:100]}")
    """

    # If a page has fewer words than this, try OCR
    MIN_WORDS_BEFORE_OCR = 20

    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr
        self._ocr_available = self._check_ocr()

    def _check_ocr(self) -> bool:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            pytesseract.get_tesseract_version()
            logger.info("Tesseract found and ready.")
            return True
        except Exception as e:
            logger.warning(f"Tesseract not found — OCR fallback disabled. {e}")
            return False

    def extract(self, pdf_path: str | Path) -> DocumentTree:
        """
        Main entry point. Extracts all pages from a PDF.

        Returns:
            DocumentTree with all PageChunks populated
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Extracting: {pdf_path.name}")

        doc = fitz.open(str(pdf_path))
        tree = DocumentTree(
            source_pdf=pdf_path.name,
            total_pages=len(doc)
        )

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_number = page_idx + 1  # 1-indexed

            chunk = self._extract_page(page, pdf_path.name, page_number)
            tree.pages.append(chunk)

            logger.debug(
                f"Page {page_number}/{len(doc)} — "
                f"{chunk.word_count} words [{chunk.extraction_method}]"
            )

        doc.close()

        # Build section structure after all pages extracted
        tree.sections = self._detect_sections(tree.pages)

        logger.info(
            f"Extracted {pdf_path.name}: "
            f"{tree.total_pages} pages, "
            f"{len(tree.sections)} sections detected"
        )
        return tree

    def _extract_page(
        self, page: fitz.Page, source_pdf: str, page_number: int
    ) -> PageChunk:
        """Extract text from a single page, with OCR fallback."""

        # Try direct text extraction first (fast)
        text = page.get_text("text")
        text = self._clean_text(text)
        word_count = len(text.split())

        method = "text"

        # Fallback to OCR if page is likely scanned
        if (
            word_count < self.MIN_WORDS_BEFORE_OCR
            and self.use_ocr
            and self._ocr_available
        ):
            ocr_text = self._ocr_page(page)
            if ocr_text and len(ocr_text.split()) > word_count:
                text = ocr_text
                word_count = len(text.split())
                method = "ocr"

        # Detect if page likely has tables
        has_tables = self._has_tables(page)

        return PageChunk(
            source_pdf=source_pdf,
            page_number=page_number,
            text=text,
            word_count=word_count,
            has_tables=has_tables,
            extraction_method=method,
        )

    def _ocr_page(self, page: fitz.Page) -> Optional[str]:
        """Render page as image and run OCR."""
        try:
            import pytesseract
            from PIL import Image
            import io

            # Render at 300 DPI for good OCR quality
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            text = pytesseract.image_to_string(img, lang="eng")
            return self._clean_text(text)
        except Exception as e:
            logger.warning(f"OCR failed on page: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text:
        - Remove excessive whitespace
        - Fix common PDF extraction artifacts
        - Preserve paragraph structure
        """
        if not text:
            return ""

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove hyphenation artifacts (word-\nbreak → wordbreak)
        text = re.sub(r"-\n(\w)", r"\1", text)

        # Collapse multiple blank lines to max 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def _has_tables(self, page: fitz.Page) -> bool:
        """
        Heuristic: detect if page likely contains tables.
        Used to flag pages where BM25 should be careful about
        text structure.
        """
        try:
            # PyMuPDF can detect table-like structures
            tables = page.find_tables()
            return len(tables.tables) > 0
        except Exception:
            return False

    def _detect_sections(self, pages: list[PageChunk]) -> list[dict]:
        """
        Detect section boundaries from text patterns.
        Legal documents typically have numbered sections,
        headings in caps, or bold-like formatting artifacts.

        Returns list of:
        {
            "title": "Section heading text",
            "start_page": 3,
            "end_page": 7,
            "summary": ""   # filled later by LLM summarizer
        }
        """
        sections = []
        current_section = None

        # Common legal document section patterns
        section_patterns = [
            r"^\d+\.\s+[A-Z][A-Za-z\s]+$",         # "1. Background Facts"
            r"^[A-Z][A-Z\s]{5,}$",                   # "BACKGROUND FACTS"
            r"^\([A-Za-z]\)\s+[A-Z]",                # "(A) The Petitioner"
            r"^WHEREAS",
            r"^NOW THEREFORE",
            r"^IN THE.*COURT",
            r"^PRAYERS?",
            r"^FACTS",
            r"^GROUNDS?",
        ]

        compiled = [re.compile(p) for p in section_patterns]

        for chunk in pages:
            lines = chunk.text.split("\n")
            for line in lines[:5]:  # Check first 5 lines of each page
                line = line.strip()
                if any(p.match(line) for p in compiled) and len(line) > 3:
                    # New section detected
                    if current_section:
                        current_section["end_page"] = chunk.page_number - 1
                        sections.append(current_section)
                    current_section = {
                        "title": line,
                        "start_page": chunk.page_number,
                        "end_page": chunk.page_number,
                        "summary": "",
                    }
                    break

        # Close last section
        if current_section and pages:
            current_section["end_page"] = pages[-1].page_number
            sections.append(current_section)

        return sections


# ── Convenience Functions ──────────────────────────────────────────────────────

def extract_pdf(pdf_path: str | Path, use_ocr: bool = True) -> DocumentTree:
    """Convenience wrapper for single PDF extraction."""
    extractor = PDFExtractor(use_ocr=use_ocr)
    return extractor.extract(pdf_path)


def extract_folder(
    folder_path: str | Path, use_ocr: bool = True
) -> list[DocumentTree]:
    """Extract all PDFs in a folder."""
    folder = Path(folder_path)
    extractor = PDFExtractor(use_ocr=use_ocr)
    trees = []

    pdfs = list(folder.glob("*.pdf"))
    logger.info(f"Found {len(pdfs)} PDFs in {folder}")

    for pdf_path in pdfs:
        try:
            tree = extractor.extract(pdf_path)
            trees.append(tree)
        except Exception as e:
            logger.error(f"Failed to extract {pdf_path.name}: {e}")

    return trees


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py path/to/file.pdf")
        sys.exit(1)

    tree = extract_pdf(sys.argv[1])
    print(f"\n{'='*60}")
    print(f"Document: {tree.source_pdf}")
    print(f"Pages: {tree.total_pages}")
    print(f"Sections detected: {len(tree.sections)}")
    print(f"\nFirst 3 pages preview:")
    for chunk in tree.pages[:3]:
        print(f"\n--- Page {chunk.page_number} [{chunk.extraction_method}] ---")
        print(chunk.text[:300])
        print("...")