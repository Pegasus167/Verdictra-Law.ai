"""
pipeline/document_registry.py
-------------------------------
Document registry for multi-file case management.

Each case can have multiple documents — petition, vakalatnama, affidavits,
transcripts, emails, orders, WhatsApp screenshots. Every document gets a
unique doc_id. The registry is the source of truth for what files belong
to a case.

Registry stored at: cases/{case_id}/document_registry.json

Structure:
{
  "case_id": "celir_llp_vs_midc",
  "documents": [
    {
      "doc_id":           "doc_001",
      "filename":         "writ_petition.pdf",
      "file_type":        "pdf",
      "uploaded_at":      "2026-04-28T10:00:00",
      "uploaded_by":      "admin",
      "status":           "processed",  # pending / processing / processed / failed
      "page_count":       87,
      "file_size_bytes":  2048000
    }
  ]
}

Every Neo4j node and relationship carries:
  sourceDocId   = "doc_001"
  sourceFilename = "writ_petition.pdf"

Every citation in an answer shows: filename + page number.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DocumentRegistry:
    """
    Manages the document registry for a case.
    Thread-safe reads. Writes should be done before background threads start.
    """

    def __init__(self, case_path: Path):
        self.case_path      = case_path
        self.registry_path  = case_path / "document_registry.json"
        self.documents_dir  = case_path / "documents"
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    # ── Registry I/O ──────────────────────────────────────────────────────────

    def load(self) -> dict:
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"documents": []}

    def save(self, registry: dict):
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    # ── Document Management ───────────────────────────────────────────────────

    def add_document(
        self,
        filename: str,
        file_size_bytes: int,
        uploaded_by: str = "system",
    ) -> str:
        """
        Register a new document and return its doc_id.
        doc_id is sequential: doc_001, doc_002, ...
        """
        registry = self.load()
        docs     = registry.get("documents", [])

        # Generate sequential doc_id
        doc_num  = len(docs) + 1
        doc_id   = f"doc_{doc_num:03d}"

        # Infer file type from extension
        ext       = Path(filename).suffix.lower().lstrip(".")
        file_type = ext if ext in {"pdf", "docx", "doc", "eml", "msg", "txt", "jpg", "jpeg", "png"} else "unknown"

        doc_entry = {
            "doc_id":          doc_id,
            "filename":        filename,
            "file_type":       file_type,
            "uploaded_at":     datetime.now().isoformat(),
            "uploaded_by":     uploaded_by,
            "status":          "pending",
            "page_count":      None,
            "file_size_bytes": file_size_bytes,
        }

        docs.append(doc_entry)
        registry["documents"] = docs
        self.save(registry)

        logger.info(f"Document registered: {doc_id} → {filename}")
        return doc_id

    def update_status(self, doc_id: str, status: str, page_count: int = None):
        """Update processing status for a document."""
        registry = self.load()
        for doc in registry.get("documents", []):
            if doc["doc_id"] == doc_id:
                doc["status"] = status
                if page_count is not None:
                    doc["page_count"] = page_count
                break
        self.save(registry)

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Get a single document entry by doc_id."""
        registry = self.load()
        for doc in registry.get("documents", []):
            if doc["doc_id"] == doc_id:
                return doc
        return None

    def get_all_documents(self) -> list[dict]:
        """Return all documents in registration order."""
        return self.load().get("documents", [])

    def get_pending_documents(self) -> list[dict]:
        """Return documents not yet processed."""
        return [d for d in self.get_all_documents() if d["status"] == "pending"]

    def all_processed(self) -> bool:
        """True if every document has been processed or failed."""
        docs = self.get_all_documents()
        if not docs:
            return False
        return all(d["status"] in {"processed", "failed"} for d in docs)

    def document_path(self, filename: str) -> Path:
        """Return the full path for a document file."""
        return self.documents_dir / filename