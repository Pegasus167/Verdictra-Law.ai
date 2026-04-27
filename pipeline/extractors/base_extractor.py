"""
pipeline/extractors/base_extractor.py
---------------------------------------
Abstract base class for all file type extractors.

Every extractor takes a file path and returns:
    - markdown_text: str   — the extracted text in Markdown format
    - page_count: int      — number of pages/sections
    - metadata: dict       — file-type specific metadata
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractionOutput:
    """Standard output from any file type extractor."""
    markdown_text: str          # Full text in Markdown format
    page_count:    int          # Number of pages or logical sections
    metadata:      dict = field(default_factory=dict)  # File-type specific extras
    warnings:      list = field(default_factory=list)  # Non-fatal issues during extraction


class BaseExtractor(ABC):
    """
    Abstract base for all file type extractors.

    Subclasses implement extract() for their specific file type.
    The router calls extract() without knowing the file type.
    """

    @abstractmethod
    def extract(self, file_path: Path, use_ocr: bool = True) -> ExtractionOutput:
        """
        Extract text from a file.

        Args:
            file_path: Path to the file on disk
            use_ocr:   Whether to use OCR for image/scanned content

        Returns:
            ExtractionOutput with markdown text, page count, metadata
        """
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """List of file extensions this extractor handles e.g. ['.pdf']"""
        ...
