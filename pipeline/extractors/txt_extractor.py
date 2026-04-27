"""
pipeline/extractors/txt_extractor.py
--------------------------------------
Plain text extractor for .txt files.

Handles:
  - WhatsApp chat exports (exported as .txt)
  - Court cause list downloads
  - Simple correspondence saved as text
  - Any plain text legal document
"""

from __future__ import annotations
import logging
from pathlib import Path

from pipeline.extractors.base_extractor import BaseExtractor, ExtractionOutput

logger = logging.getLogger(__name__)


class TXTExtractor(BaseExtractor):

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt"]

    def extract(self, file_path: Path, use_ocr: bool = True) -> ExtractionOutput:
        try:
            # Try UTF-8 first, fall back to latin-1
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = file_path.read_text(encoding="latin-1")

            # Estimate page count
            page_count = max(1, len(text.split()) // 300)

            # Wrap in minimal Markdown
            markdown_text = f"# {file_path.stem}\n\n{text}"

            logger.info(f"TXT extracted: {file_path.name} ({len(text.split())} words)")

            return ExtractionOutput(
                markdown_text=markdown_text,
                page_count=page_count,
                metadata={
                    "source_file": file_path.name,
                    "extractor":   "plain-text",
                    "word_count":  len(text.split()),
                },
            )

        except Exception as e:
            logger.error(f"TXT extraction failed for {file_path.name}: {e}")
            raise
