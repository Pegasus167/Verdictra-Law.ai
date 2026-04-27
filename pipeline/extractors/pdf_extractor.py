"""
pipeline/extractors/pdf_extractor.py
--------------------------------------
PDF extractor — wraps the existing pdf_to_markdown pipeline.

Uses pdfplumber for digital PDFs and Tesseract OCR for scanned pages.
This is the primary extractor for Indian legal documents.
"""

from __future__ import annotations
import logging
from pathlib import Path

from pipeline.extractors.base_extractor import BaseExtractor, ExtractionOutput

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def extract(self, file_path: Path, use_ocr: bool = True) -> ExtractionOutput:
        from pipeline.pdf_to_markdown import pdf_to_markdown

        # Use a temp output path in the same directory
        md_output = file_path.with_suffix(".md")

        try:
            md_text = pdf_to_markdown(file_path, md_output, use_ocr=use_ocr)

            # Count pages from markdown — pdf_to_markdown inserts page markers
            page_count = md_text.count("<!-- page") or md_text.count("\n# Page ")
            if page_count == 0:
                # Fallback: estimate from word count
                page_count = max(1, len(md_text.split()) // 300)

            return ExtractionOutput(
                markdown_text=md_text,
                page_count=page_count,
                metadata={"source_file": file_path.name, "extractor": "pdfplumber+tesseract"},
            )

        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path.name}: {e}")
            raise
