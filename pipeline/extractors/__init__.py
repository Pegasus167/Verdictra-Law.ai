"""
pipeline/extractors/
----------------------
File type extractors for Verdictra multi-document ingestion.

Public API — import from here:
    from pipeline.extractors import extract_file, is_supported, supported_extensions
"""

from pipeline.extractors.router import extract_file, is_supported, supported_extensions, get_extractor

__all__ = ["extract_file", "is_supported", "supported_extensions", "get_extractor"]
