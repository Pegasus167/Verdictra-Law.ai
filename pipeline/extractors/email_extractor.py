"""
pipeline/extractors/email_extractor.py
----------------------------------------
Email extractor using Python's built-in email library.

Handles .eml files (standard email format).
Extracts structured metadata as first-class entities:
  - From/To/CC → Person entities with SENT_TO relationships
  - Date → Event entity
  - Subject → context for all entities in body
  - Body → run GLiNER on this

Why emails matter in Indian litigation:
  - Demand notices often sent by email
  - Correspondence between parties is evidence
  - Client instructions and admissions appear in emails
  - WhatsApp message exports can be saved as .txt or .eml
"""

from __future__ import annotations
import email
import logging
from pathlib import Path

from pipeline.extractors.base_extractor import BaseExtractor, ExtractionOutput

logger = logging.getLogger(__name__)


class EmailExtractor(BaseExtractor):

    @property
    def supported_extensions(self) -> list[str]:
        return [".eml", ".msg"]

    def extract(self, file_path: Path, use_ocr: bool = True) -> ExtractionOutput:
        try:
            with open(file_path, "rb") as f:
                msg = email.message_from_bytes(f.read())

            # Extract structured metadata
            sender     = msg.get("From", "Unknown Sender")
            recipients = msg.get("To", "")
            cc         = msg.get("Cc", "")
            date_str   = msg.get("Date", "")
            subject    = msg.get("Subject", "No Subject")

            # Extract body text
            body_parts = []
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        try:
                            body_parts.append(
                                part.get_payload(decode=True).decode("utf-8", errors="replace")
                            )
                        except Exception:
                            pass
            else:
                try:
                    body_parts.append(
                        msg.get_payload(decode=True).decode("utf-8", errors="replace")
                    )
                except Exception:
                    body_parts.append(str(msg.get_payload()))

            body = "\n".join(body_parts)

            # Build structured Markdown
            # The structured header becomes queryable context for GLiNER
            lines = [
                f"# Email: {subject}",
                "",
                f"**From:** {sender}",
                f"**To:** {recipients}",
            ]
            if cc:
                lines.append(f"**CC:** {cc}")
            lines.extend([
                f"**Date:** {date_str}",
                f"**Subject:** {subject}",
                "",
                "## Email Body",
                "",
                body,
            ])

            markdown_text = "\n".join(lines)

            logger.info(f"Email extracted: {file_path.name} — Subject: {subject}")

            return ExtractionOutput(
                markdown_text=markdown_text,
                page_count=1,  # Emails are one logical document
                metadata={
                    "source_file": file_path.name,
                    "extractor":   "python-email",
                    "from":        sender,
                    "to":          recipients,
                    "date":        date_str,
                    "subject":     subject,
                },
            )

        except Exception as e:
            logger.error(f"Email extraction failed for {file_path.name}: {e}")
            raise
