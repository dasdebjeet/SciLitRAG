
"""
Document upload endpoints for SciLitRAG.

Uploads are validated and stored independently from the indexed arXiv
corpus. This router does not initialize retrieval or language models.
"""

from __future__ import annotations

import logging
from typing import Annotated

import pymupdf
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from scilitrag.api.schemas.response import UploadResponse
from scilitrag.core.upload_storage import (
    MAX_UPLOAD_SIZE_BYTES,
    store_uploaded_pdf,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter()

# Read incrementally to enforce the upload limit before buffering the
# entire request body in application memory.
UPLOAD_READ_CHUNK_SIZE = 1024 * 1024


@router.post(
    "/uploads",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a research paper",
    description=(
        "Upload a PDF research paper for isolated, document-scoped "
        "question answering."
    ),
)
async def upload_research_paper(
    file: Annotated[UploadFile, File()],
) -> UploadResponse:
    """Validate and persist one uploaded research paper."""

    filename = file.filename or ""

    if not filename.lower().endswith(".pdf"):
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    content = bytearray()

    try:
        # Read bounded chunks instead of loading an arbitrarily large
        # multipart upload into memory.
        while True:
            chunk = await file.read(UPLOAD_READ_CHUNK_SIZE)

            if not chunk:
                break

            if len(content) + len(chunk) > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="The uploaded PDF exceeds the 25 MB limit.",
                )

            content.extend(chunk)

        document = store_uploaded_pdf(
            content=bytes(content),
            filename=filename,
        )

        return UploadResponse.model_validate(
            {
                "upload_id": document.upload_id,
                "filename": document.filename,
                "title": document.title,
                "page_count": document.page_count,
                "total_characters": document.total_characters,
                "created_at": document.created_at,
            }
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except pymupdf.FileDataError as exc:
        # Malformed PDFs may fail during PyMuPDF parsing rather than
        # during the initial PDF-header validation.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded PDF could not be processed.",
        ) from exc

    except OSError as exc:
        LOGGER.exception("Failed to persist an uploaded research PDF.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The uploaded PDF could not be stored.",
        ) from exc

    finally:
        await file.close()
