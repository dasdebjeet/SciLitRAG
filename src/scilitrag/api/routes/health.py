
"""
Health-check endpoints for the SciLitRAG API.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from qdrant_client import QdrantClient

from scilitrag.core.paths import DEFAULT_CHUNKS_PATH


LOGGER = logging.getLogger(__name__)

router = APIRouter()

QDRANT_URL = "http://localhost:16333"


@router.get(
    "/health",
    summary="Check API liveness",
)
def health_check() -> dict[str, str]:
    """Confirm that the API process is running."""

    return {
        "status": "ok",
        "service": "scilitrag-api",
    }


@router.get(
    "/health/ready",
    summary="Check API readiness",
)
def readiness_check() -> dict[str, str]:
    """Check whether essential backend resources are accessible."""

    if not DEFAULT_CHUNKS_PATH.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The processed chunks file is unavailable.",
        )

    client = None

    try:
        client = QdrantClient(url=QDRANT_URL)

        # A lightweight service-level check; this does not load
        # the embedding model or perform a vector search.
        client.get_collections()

        return {
            "status": "ready",
            "qdrant": "connected",
            "chunks": "available",
        }

    except Exception as exc:
        LOGGER.exception("Readiness check failed.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="One or more backend dependencies are unavailable.",
        ) from exc

    finally:
        if client is not None:
            client.close()
