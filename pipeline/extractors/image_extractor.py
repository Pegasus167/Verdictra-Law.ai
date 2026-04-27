"""
pipeline/extractors/image_extractor.py
----------------------------------------
Image extractor using Tesseract OCR directly.

Handles: .jpg, .jpeg, .png, .tiff, .bmp

Use cases:
  - WhatsApp screenshots (evidence in Indian courts)
  - Scanned documents saved as images (not PDF)
  - Photos of physical documents
  - Court notice photographs

Note: Image quality heavily affects OCR accuracy.
High-resolution images (>300 DPI) give much better results.
"""

from __future__ import annotations
import logging
from pathlib import Path

from pipeline.extractors.base_extractor import BaseExtractor, ExtractionOutput

logger = logging.getLogger(__name__)


class ImageExtractor(BaseExtractor):

    @property
    def supported_extensions(self) -> list[str]:
        return [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"]

    def extract(self, file_path: Path, use_ocr: bool = True) -> ExtractionOutput:
        warnings = []

        try:
            import pytesseract
            from PIL import Image

            img  = Image.open(str(file_path))

            # Check image size — warn if too small for reliable OCR
            width, height = img.size
            if width < 800 or height < 600:
                warnings.append(
                    f"Image {file_path.name} is small ({width}x{height}px). "
                    "OCR quality may be low. Higher resolution images give better results."
                )

            # Run Tesseract OCR
            text = pytesseract.image_to_string(img, lang="eng")

            if not text.strip():
                warnings.append(f"No text extracted from {file_path.name} — image may be blank or unreadable.")

            markdown_text = f"# Image: {file_path.name}\n\n{text}"

            logger.info(
                f"Image OCR: {file_path.name} ({width}x{height}px, "
                f"{len(text.split())} words extracted)"
            )

            return ExtractionOutput(
                markdown_text=markdown_text,
                page_count=1,
                metadata={
                    "source_file":  file_path.name,
                    "extractor":    "tesseract-ocr",
                    "image_size":   f"{width}x{height}",
                    "word_count":   len(text.split()),
                },
                warnings=warnings,
            )

        except ImportError:
            raise RuntimeError(
                "pytesseract or Pillow not installed. "
                "Run: poetry add pytesseract Pillow"
            )
        except Exception as e:
            logger.error(f"Image extraction failed for {file_path.name}: {e}")
            raise
