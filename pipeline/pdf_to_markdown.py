"""
pipeline/pdf_to_markdown.py
----------------------------
Stage 1 — Convert PDF to clean Markdown with legal-specific heading detection.

Flow:
    PDF
     ├─ Digital pages  → pdfplumber (perfect text + table extraction, free)
     └─ Scanned pages  → pdf2image → Tesseract OCR (parallel, ProcessPoolExecutor)
     ↓
    Raw text per page
     ↓
    detect_legal_heading() → Markdown with # / ## / ### / #### markers
     ↓
    Single .md file saved to cases/{case_id}/{case_id}.md

Why pdfplumber over PyMuPDF for digital PDFs:
    - Handles multi-column layouts correctly
    - Extracts tables as structured text not garbled characters
    - Better at preserving reading order across columns
    - Confirmed superior for Indian court documents and annual reports
    - Free, no API cost

Why parallel OCR:
    - Each page is independent — embarrassingly parallel
    - ProcessPoolExecutor with 4 workers reduces OCR from 10 mins to ~2-3 mins
    - Workers = min(4, cpu_count) so it scales to available hardware

Detection strategy:
    1. Try pdfplumber first — if page has >= MIN_WORDS_FOR_TEXT words, it's digital
    2. If not enough text → page is scanned → queue for parallel Tesseract OCR
    3. Merge results preserving page order
"""

from __future__ import annotations

import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

# Tesseract path — Windows default, override via env
TESSERACT_PATH = os.environ.get(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Min words before treating page as scanned
MIN_WORDS_FOR_TEXT = 20

# Max parallel OCR workers
MAX_OCR_WORKERS = min(4, os.cpu_count() or 2)


# ── Legal heading patterns ─────────────────────────────────────────────────────

_TITLE_PATTERNS = [
    re.compile(r"^IN THE (HON'?BLE |HONOURABLE )?(.+COURT|TRIBUNAL)", re.IGNORECASE),
    re.compile(r"^WRIT PETITION", re.IGNORECASE),
    re.compile(r"^CIVIL APPEAL", re.IGNORECASE),
    re.compile(r"^SPECIAL LEAVE PETITION", re.IGNORECASE),
    re.compile(r"^PETITION UNDER", re.IGNORECASE),
    re.compile(r"^BETWEEN\s*:?$", re.IGNORECASE),
]

_SECTION_PATTERNS = [
    re.compile(r"^(FACTS|BACKGROUND FACTS?|BRIEF FACTS?)\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^GROUNDS?\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^PRAYERS?\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^RELIEF(S)? SOUGHT\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^SYNOPSIS\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^LIST OF DATES\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^INDEX\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^ANNEXURES?\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^AFFIDAVIT\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^SUBMISSIONS?\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^ARGUMENTS?\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^ORDERS?\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^JUDGMENT\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^WHEREAS\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^NOW THEREFORE\s*:?\s*$", re.IGNORECASE),
]

_SUBSECTION_PATTERNS = [
    re.compile(r"^\([A-Z]\)\s+\S"),
    re.compile(r"^\d+\.\s+[A-Z]"),
    re.compile(r"^[IVX]+\.\s+[A-Z]"),
    re.compile(r"^(CLAUSE|SECTION|ARTICLE)\s+\d+", re.IGNORECASE),
]

_ITEM_PATTERNS = [
    re.compile(r"^\([a-z]\)\s+\S"),
    re.compile(r"^\([ivxlc]+\)\s+\S"),
    re.compile(r"^\d+\.\d+\s+\S"),
]


def detect_legal_heading(line: str) -> str:
    """Convert a line of legal text into Markdown heading or plain text."""
    stripped = line.strip()
    if not stripped:
        return ""

    for pattern in _TITLE_PATTERNS:
        if pattern.match(stripped):
            return f"# {stripped}"

    for pattern in _SECTION_PATTERNS:
        if pattern.match(stripped):
            return f"## {stripped}"

    words = stripped.split()
    if (
        stripped.isupper()
        and 2 <= len(words) <= 8
        and not any(c.isdigit() for c in stripped[:3])
    ):
        return f"## {stripped}"

    for pattern in _SUBSECTION_PATTERNS:
        if pattern.match(stripped):
            return f"### {stripped}"

    for pattern in _ITEM_PATTERNS:
        if pattern.match(stripped):
            return f"#### {stripped}"

    return stripped


# ── Text cleaning ──────────────────────────────────────────────────────────────

def _clean_raw_text(text: str) -> str:
    """Clean OCR/extracted text artifacts."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _text_to_markdown(raw: str) -> str:
    """Apply heading detection to raw text and return Markdown string."""
    md_lines = []
    for line in raw.split("\n"):
        md_lines.append(detect_legal_heading(line))
    return "\n".join(md_lines)


# ── pdfplumber extraction (digital PDFs) ───────────────────────────────────────

def _extract_page_pdfplumber(pdf_path: str, page_idx: int) -> str:
    """
    Extract text from a single page using pdfplumber.
    Handles multi-column layouts and tables better than PyMuPDF.

    Returns empty string if pdfplumber not available or extraction fails.
    """
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_idx]

            # Extract tables first — convert to text representation
            table_texts = []
            for table in page.extract_tables() or []:
                rows = []
                for row in table:
                    clean_row = [str(cell or "").strip() for cell in row]
                    if any(clean_row):
                        rows.append(" | ".join(clean_row))
                if rows:
                    table_texts.append("\n".join(rows))

            # Extract main text
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

            # Append table text if not already in main text
            for t in table_texts:
                if t[:30] not in text:
                    text = text + "\n\n" + t

            return text

    except ImportError:
        return ""
    except Exception as e:
        logger.debug(f"pdfplumber failed on page {page_idx + 1}: {e}")
        return ""


# ── Tesseract OCR (scanned pages, runs in worker process) ─────────────────────

def _ocr_single_page(args: tuple) -> tuple[int, str]:
    """
    OCR a single page. Designed to run in a worker process.

    Args:
        args: (pdf_path, page_idx, tesseract_path)

    Returns:
        (page_idx, ocr_text)
    """
    pdf_path, page_idx, tesseract_path = args
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io

        pytesseract.pytesseract.tesseract_cmd = tesseract_path

        doc = fitz.open(pdf_path)
        page = doc[page_idx]

        # 300 DPI for good OCR accuracy
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        doc.close()

        text = pytesseract.image_to_string(img, lang="eng")
        return page_idx, text

    except Exception as e:
        logger.debug(f"Tesseract OCR failed on page {page_idx + 1}: {e}")
        return page_idx, ""


# ── Main converter ─────────────────────────────────────────────────────────────

def pdf_to_markdown(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
    use_ocr: bool = True,
) -> str:
    """
    Convert a PDF to Markdown with legal-specific heading detection.

    Strategy:
        1. Try pdfplumber on every page (free, handles digital PDFs perfectly)
        2. Pages with < MIN_WORDS_FOR_TEXT words → scanned → parallel Tesseract OCR
        3. Merge results preserving page order
        4. Apply heading detection to produce Markdown

    Args:
        pdf_path:    Path to input PDF
        output_path: Where to save .md file
        use_ocr:     Whether to use Tesseract for scanned pages

    Returns:
        Full Markdown string
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Converting {pdf_path.name} to Markdown...")

    # ── Phase 1: pdfplumber pass on all pages ──────────────────────────────────
    import fitz
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    doc.close()

    logger.info(f"  Phase 1: pdfplumber extraction ({total_pages} pages)...")

    page_texts: dict[int, str] = {}
    scanned_pages: list[int]   = []

    for page_idx in range(total_pages):
        text = _extract_page_pdfplumber(str(pdf_path), page_idx)
        text = _clean_raw_text(text)
        word_count = len(text.split())

        if word_count >= MIN_WORDS_FOR_TEXT:
            page_texts[page_idx] = text
        else:
            # Not enough text — mark as scanned for OCR
            scanned_pages.append(page_idx)
            page_texts[page_idx] = text  # keep whatever we got as fallback

    digital_count = total_pages - len(scanned_pages)
    logger.info(
        f"  pdfplumber: {digital_count} digital pages, "
        f"{len(scanned_pages)} scanned pages queued for OCR"
    )

    # ── Phase 2: parallel Tesseract OCR for scanned pages ─────────────────────
    if scanned_pages and use_ocr:
        logger.info(
            f"  Phase 2: parallel OCR on {len(scanned_pages)} pages "
            f"({MAX_OCR_WORKERS} workers)..."
        )

        ocr_args = [
            (str(pdf_path), page_idx, TESSERACT_PATH)
            for page_idx in scanned_pages
        ]

        ocr_results: dict[int, str] = {}

        with ProcessPoolExecutor(max_workers=MAX_OCR_WORKERS) as executor:
            futures = {
                executor.submit(_ocr_single_page, args): args[1]
                for args in ocr_args
            }
            completed = 0
            for future in as_completed(futures):
                page_idx, ocr_text = future.result()
                ocr_text = _clean_raw_text(ocr_text)

                # Only use OCR if it gives more text than pdfplumber
                if len(ocr_text.split()) > len(page_texts.get(page_idx, "").split()):
                    ocr_results[page_idx] = ocr_text
                else:
                    ocr_results[page_idx] = page_texts.get(page_idx, "")

                completed += 1
                if completed % 10 == 0:
                    logger.info(f"  OCR progress: {completed}/{len(scanned_pages)} pages")

        # Merge OCR results
        page_texts.update(ocr_results)
        logger.info(f"  OCR complete: {len(ocr_results)} pages processed")

    elif scanned_pages and not use_ocr:
        logger.info(
            f"  OCR disabled — {len(scanned_pages)} scanned pages will have limited text"
        )

    # ── Phase 3: apply heading detection and build Markdown ───────────────────
    logger.info("  Phase 3: applying heading detection...")

    md_pages = []
    for page_idx in sorted(page_texts.keys()):
        raw = page_texts[page_idx]
        if not raw.strip():
            continue
        page_num = page_idx + 1
        page_md  = _text_to_markdown(raw)
        md_pages.append(f"<!-- page:{page_num} -->\n{page_md}")

    full_md = "\n\n".join(md_pages)

    # ── Save ───────────────────────────────────────────────────────────────────
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_md)
        logger.info(
            f"Markdown saved → {output_path} "
            f"({len(md_pages)} pages, {len(full_md):,} chars)"
        )

    return full_md


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_markdown.py <pdf_path> [output.md]")
        sys.exit(1)

    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else pdf.replace(".pdf", ".md")
    md  = pdf_to_markdown(pdf, out)
    print(f"\nFirst 500 chars:\n{md[:500]}")