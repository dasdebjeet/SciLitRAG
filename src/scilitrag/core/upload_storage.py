
"""
Isolated storage and PDF extraction for user-uploaded research papers.

Uploaded documents are stored independently from the indexed arXiv corpus.
This module deliberately does not create chunks, embeddings, or Qdrant
records; those belong to the subsequent document-local retrieval stage.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf

from scilitrag.core.paths import UPLOADS_DIR

LOGGER = logging.getLogger(__name__)

MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_PAGES = 200


@dataclass(frozen=True)
class UploadedDocument:
    """Public metadata describing a successfully stored PDF."""

    upload_id: str
    filename: str
    title: str
    page_count: int
    total_characters: int
    created_at: str


def _safe_filename(filename: str | None) -> str:
    """Normalize a client-provided filename for metadata display only."""

    if not filename:
        return "uploaded-paper.pdf"

    # Remove directory components and control characters. The original
    # filename is never used to construct a filesystem path.
    normalized = Path(filename.replace("\\", "/")).name
    normalized = re.sub(r"[\x00-\x1f\x7f]", "", normalized).strip()

    if not normalized:
        return "uploaded-paper.pdf"

    return normalized[:255]


def store_uploaded_pdf(
    content: bytes,
    filename: str | None,
    *,
    uploads_dir: Path = UPLOADS_DIR,
    max_size_bytes: int = MAX_UPLOAD_SIZE_BYTES,
    max_pages: int = MAX_UPLOAD_PAGES,
) -> UploadedDocument:
    """
    Validate, extract, and atomically store an uploaded research PDF.

    Args:
        content: Complete PDF contents received from the upload endpoint.
        filename: Original client-provided filename, used only as metadata.
        uploads_dir: Root directory for isolated user uploads.
        max_size_bytes: Maximum accepted PDF size in bytes.
        max_pages: Maximum number of pages accepted per document.

    Returns:
        Metadata for the stored document, including its opaque upload ID.

    Raises:
        ValueError: If the file is invalid, encrypted, empty, too large,
            or exceeds the configured page limit.
        OSError: If the document cannot be persisted.
    """

    if not content:
        raise ValueError("The uploaded PDF is empty.")

    if len(content) > max_size_bytes:
        raise ValueError("The uploaded PDF exceeds the maximum file size.")

    # A PDF header is expected near the beginning of a conforming file.
    # PyMuPDF performs the deeper structural validation below.
    if b"%PDF-" not in content[:1024]:
        raise ValueError("The uploaded file is not a valid PDF.")

    original_filename = _safe_filename(filename)

    if Path(original_filename).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    upload_id = str(uuid.uuid4())
    uploads_dir = uploads_dir.resolve()
    uploads_dir.mkdir(parents=True, exist_ok=True)

    staging_dir = uploads_dir / f".tmp-{upload_id}"
    document_dir = uploads_dir / upload_id

    try:
        staging_dir.mkdir()

        # Parse directly from memory. This avoids writing unvalidated
        # client data to its final storage location.
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")

            page_count = len(document)

            if page_count == 0:
                raise ValueError("The PDF does not contain any pages.")

            if page_count > max_pages:
                raise ValueError(
                    f"The PDF exceeds the maximum of {max_pages} pages."
                )

            pdf_metadata = document.metadata or {}
            title = (
                str(pdf_metadata.get("title") or "").strip()
                or Path(original_filename).stem
            )

            pages: list[dict[str, Any]] = []
            total_characters = 0

            for page_index, page in enumerate(document):
                extracted_text = page.get_text("text", sort=True).strip()
                total_characters += len(extracted_text)

                pages.append(
                    {
                        "page_number": page_index + 1,
                        "total_pages": page_count,
                        "text": extracted_text,
                        "character_count": len(extracted_text),
                    }
                )

            # Image-only/scanned PDFs require OCR, which is intentionally
            # outside this text-extraction MVP.
            if total_characters == 0:
                raise ValueError(
                    "No extractable text was found in the PDF. "
                    "Scanned PDFs are not supported yet."
                )

        created_at = datetime.now(UTC).isoformat()

        metadata = UploadedDocument(
            upload_id=upload_id,
            filename=original_filename,
            title=title[:500],
            page_count=page_count,
            total_characters=total_characters,
            created_at=created_at,
        )

        # Persist the original document and extracted page records inside
        # the staging directory before making the upload visible.
        (staging_dir / "document.pdf").write_bytes(content)

        with (staging_dir / "pages.json").open(
            "w", encoding="utf-8"
        ) as pages_file:
            json.dump(pages, pages_file, ensure_ascii=False)

        with (staging_dir / "metadata.json").open(
            "w", encoding="utf-8"
        ) as metadata_file:
            json.dump(asdict(metadata), metadata_file, ensure_ascii=False)

        # The final directory becomes visible only after all files have
        # been written successfully.
        staging_dir.rename(document_dir)

        LOGGER.info(
            "Stored uploaded PDF %s (%d pages, %d characters).",
            upload_id,
            page_count,
            total_characters,
        )

        return metadata

    except Exception:
        # Never leave a partially processed upload in the storage area.
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

        raise
