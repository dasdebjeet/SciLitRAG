
"""
Research question endpoint for SciLitRAG.

Exposes the existing answer pipeline through a REST API without
moving retrieval, reranking, generation, or citation logic into
the HTTP layer.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from scilitrag.api.dependencies import (
    get_answer_pipeline,
    get_inference_lock,
)
from scilitrag.api.schemas.request import AskRequest
from scilitrag.api.schemas.response import (
    AskResponse,
    CitationResponse,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer a research question",
    description=(
        "Retrieve relevant research passages, rerank them, "
        "and generate an evidence-grounded answer."
    ),
)
def ask_question(
    request: AskRequest,
    pipeline: Annotated[
        Any,
        Depends(get_answer_pipeline),
    ],
    inference_lock: Annotated[
        Lock,
        Depends(get_inference_lock),
    ],
) -> AskResponse:
    """Generate an evidence-grounded answer for a research question."""

    try:
        # Serialize the complete model lifecycle to prevent concurrent
        # requests from competing for GPU memory.
        with inference_lock:
            result = pipeline.answer(
                question=request.question,
                arxiv_id=request.arxiv_id,
            )

        # Convert internal Evidence objects into API response models.
        citations = [
            CitationResponse(
                evidence_id=item.evidence_id,
                chunk_id=item.chunk_id,
                arxiv_id=item.arxiv_id,
                title=item.title,
                page_number=item.page_number,
                text=item.text,
                reranker_score=item.reranker_score,
            )
            for item in result.citations
        ]

        return AskResponse(
            question=result.question,
            answer=result.answer,
            citations=citations,
            context_token_count=result.context_token_count,
        )

    except Exception as exc:
        LOGGER.exception(
            "Failed to process research question: %s",
            request.question,
        )

        # Do not expose internal exception details or stack traces to clients.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The research question could not be processed.",
        ) from exc
