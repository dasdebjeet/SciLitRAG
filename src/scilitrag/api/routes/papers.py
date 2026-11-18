
"""
Paper library endpoint for SciLitRAG.

Reads paper metadata from the local metadata catalogue and exposes
a validated, read-only collection to frontend applications.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from scilitrag.api.schemas.response import PaperResponse
from scilitrag.core.paths import PAPERS_METADATA_PATH

LOGGER = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/papers",
    response_model=list[PaperResponse],
    status_code=status.HTTP_200_OK,
    summary="List available research papers",
    description=(
        "Return the research papers registered in the local "
        "SciLitRAG metadata catalogue."
    ),
)
def get_papers() -> list[PaperResponse]:
    """Load and validate all paper records from the metadata catalogue."""

    try:
        # Read the catalogue on each request so newly added papers become
        # visible without restarting the API process.
        with PAPERS_METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
            records = json.load(metadata_file)

        # The catalogue must contain a JSON array of paper records.
        if not isinstance(records, list):
            raise ValueError("Paper metadata must be a JSON array.")

        # Validate every record against the public API response contract.
        # Extra internal metadata is ignored, but missing or invalid public
        # fields cause the entire catalogue request to fail.
        return [
            PaperResponse.model_validate(
                {
                    field: record.get(field)
                    for field in (
                        "arxiv_id",
                        "title",
                        "authors",
                        "abstract",
                        "published",
                        "categories",
                        "primary_category",
                        "pdf_url",
                    )
                }
            )
            for record in records
        ]

    except (OSError, ValueError, ValidationError) as exc:
        LOGGER.exception(
            "Failed to load or validate the paper metadata catalogue."
        )

        # Keep filesystem paths and validation internals out of API responses.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The paper library could not be loaded.",
        ) from exc
