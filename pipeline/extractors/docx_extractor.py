"""
pipeline/extractors/docx_extractor.py
---------------------------------------
DOCX extractor using python-docx.

Preserves heading hierarchy — headings become Markdown headers.
This gives the tree_builder proper section structure to work with.

Better than PDF for:
  - Transcripts (structured paragraphs, speaker labels)
  - Affidavits (typed, clean structure)
  - Correspondence (formal letter structure)
  - Any document that was originally created in Word
"""

from __future__ import annotations
import logging
from pathlib import Path

from pipeline.extractors.base_extractor import BaseExtractor, ExtractionOutput

logger = logging.getLogger(__name__)


class DOCXExtractor(BaseExtractor):

    @property
    def supported_extensions(self) -> list[str]:
        return [".docx", ".doc"]

    def extract(self, file_path: Path, use_ocr: bool = True) -> ExtractionOutput:
        try:
            from docx import Document
        except ImportError:
            raise RuntimeError("python-docx not installed. Run: poetry add python-docx")

        try:
            doc = Document(str(file_path))
            lines = []
            page_count = 1  # DOCX has no page concept — count sections instead
            section_count = 0

            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue

                style_name = para.style.name.lower() if para.style else ""

                if "heading 1" in style_name:
                    lines.append(f"\n# {text}\n")
                    section_count += 1
                elif "heading 2" in style_name:
                    lines.append(f"\n## {text}\n")
                    section_count += 1
                elif "heading 3" in style_name:
                    lines.append(f"\n### {text}\n")
                elif "heading" in style_name:
                    lines.append(f"\n#### {text}\n")
                else:
                    lines.append(text)

            # Also extract text from tables
            for table in doc.tables:
                lines.append("\n")
                for row in table.rows:
                    row_text = " | ".join(
                        cell.text.strip() for cell in row.cells if cell.text.strip()
                    )
                    if row_text:
                        lines.append(row_text)
                lines.append("\n")

            markdown_text = "\n".join(lines)

            # Use section count as page proxy if meaningful
            if section_count > 0:
                page_count = section_count
            else:
                # Estimate from word count (avg 300 words/page)
                page_count = max(1, len(markdown_text.split()) // 300)

            logger.info(
                f"DOCX extracted: {file_path.name} "
                f"({len(doc.paragraphs)} paragraphs, ~{page_count} sections)"
            )

            return ExtractionOutput(
                markdown_text=markdown_text,
                page_count=page_count,
                metadata={
                    "source_file":   file_path.name,
                    "extractor":     "python-docx",
                    "paragraph_count": len(doc.paragraphs),
                    "table_count":   len(doc.tables),
                },
            )

        except Exception as e:
            logger.error(f"DOCX extraction failed for {file_path.name}: {e}")
            raise
