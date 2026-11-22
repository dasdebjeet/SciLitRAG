
"""
Research question endpoint for SciLitRAG.

Exposes the existing answer pipeline through a REST API without
moving retrieval, reranking, generation, or citation logic into
the HTTP layer.
"""

from __future__ import annotations

import logging
import time
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

    request_started = time.perf_counter()

    try:
        # Serialize inference while the shared GPU model lifecycle is
        # protected by the application-level inference lock.
        with inference_lock:
            result = pipeline.answer(
                question=request.question,
                arxiv_id=request.arxiv_id,
                arxiv_ids=request.arxiv_ids,
                history=[
                    message.model_dump()
                    for message in request.history
                ],
            )

        # Convert internal evidence records into API response models.
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

        # Include pipeline execution and API response construction.
        # This is server-side processing time, not browser/network time.
        total_duration = time.perf_counter() - request_started

        LOGGER.info(
            "Research request completed | duration=%.3fs | "
            "citations=%d | context_tokens=%d | generated_tokens=%s",
            total_duration,
            len(citations),
            result.context_token_count,
            getattr(result, "generation_token_count", None),
        )

        return AskResponse(
            question=result.question,
            answer=result.answer,
            citations=citations,
            context_token_count=result.context_token_count,
            generation_duration_seconds=getattr(
                result,
                "generation_duration_seconds",
                None,
            ),
            generation_token_count=getattr(
                result,
                "generation_token_count",
                None,
            ),
            tokens_per_second=getattr(
                result,
                "tokens_per_second",
                None,
            ),
            total_response_duration_seconds=total_duration,
        )

    except Exception:
        # Do not put user prompts or retrieved paper content in error logs.
        LOGGER.exception(
            "Research request failed | elapsed=%.3fs",
            time.perf_counter() - request_started,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The research question could not be processed.",
        )
