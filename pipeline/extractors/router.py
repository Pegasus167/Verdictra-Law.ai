"""
pipeline/extractors/router.py
-------------------------------
File type router — routes each file to the correct extractor.

This is the single entry point for all file extraction.
The ingestion pipeline calls extract_file() without knowing the file type.

Supported formats:
    .pdf            → PDFExtractor   (pdfplumber + Tesseract)
    .docx / .doc    → DOCXExtractor  (python-docx)
    .eml / .msg     → EmailExtractor (python email library)
    .jpg/.jpeg/.png → ImageExtractor (Tesseract OCR)
    .txt            → TXTExtractor   (plain text)

Adding a new format:
    1. Create a new extractor in pipeline/extractors/
    2. Import and register it in EXTRACTORS list below
"""

from __future__ import annotations
import logging
from pathlib import Path

from pipeline.extractors.base_extractor import BaseExtractor, ExtractionOutput
from pipeline.extractors.pdf_extractor   import PDFExtractor
from pipeline.extractors.docx_extractor  import DOCXExtractor
from pipeline.extractors.email_extractor import EmailExtractor
from pipeline.extractors.image_extractor import ImageExtractor
from pipeline.extractors.txt_extractor   import TXTExtractor

logger = logging.getLogger(__name__)

# ── Registered extractors ──────────────────────────────────────────────────────
# Order matters for overlapping extensions (first match wins)
_EXTRACTORS: list[BaseExtractor] = [
    PDFExtractor(),
    DOCXExtractor(),
    EmailExtractor(),
    ImageExtractor(),
    TXTExtractor(),
]

# Build extension → extractor lookup
_EXTENSION_MAP: dict[str, BaseExtractor] = {}
for _extractor in _EXTRACTORS:
    for _ext in _extractor.supported_extensions:
        _EXTENSION_MAP[_ext.lower()] = _extractor


def get_extractor(file_path: Path) -> BaseExtractor:
    """
    Return the appropriate extractor for this file type.
    Raises ValueError if no extractor supports the extension.
    """
    ext = file_path.suffix.lower()
    if ext not in _EXTENSION_MAP:
        supported = sorted(_EXTENSION_MAP.keys())
        raise ValueError(
            f"No extractor for '{ext}' files. "
            f"Supported formats: {', '.join(supported)}"
        )
    return _EXTENSION_MAP[ext]


def extract_file(
    file_path: Path,
    use_ocr: bool = True,
) -> ExtractionOutput:
    """
    Extract text from any supported file type.

    Args:
        file_path: Path to the file
        use_ocr:   Whether to use OCR for scanned content

    Returns:
        ExtractionOutput with markdown text, page count, metadata

    Raises:
        ValueError: if file type is not supported
        Various extraction errors if the file is corrupt or unreadable
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extractor = get_extractor(file_path)
    ext       = file_path.suffix.lower()

    logger.info(f"Extracting {file_path.name} using {extractor.__class__.__name__}")

    output = extractor.extract(file_path, use_ocr=use_ocr)

    if output.warnings:
        for warning in output.warnings:
            logger.warning(f"[{file_path.name}] {warning}")

    logger.info(
        f"Extraction complete: {file_path.name} "
        f"({output.page_count} pages, {len(output.markdown_text.split())} words)"
    )

    return output


def is_supported(file_path: Path) -> bool:
    """Return True if this file type has an extractor."""
    return file_path.suffix.lower() in _EXTENSION_MAP


def supported_extensions() -> list[str]:
    """Return all supported file extensions."""
    return sorted(_EXTENSION_MAP.keys())
