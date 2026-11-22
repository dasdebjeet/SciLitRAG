
"""
Paper comparison endpoint for SciLitRAG.

Exposes the multi-paper comparison pipeline through a REST API while
keeping retrieval, reranking, context assembly, generation, and citation
resolution inside the pipeline layer.
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
from scilitrag.api.schemas.request import CompareRequest
from scilitrag.api.schemas.response import (
    CitationResponse,
    CompareResponse,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/compare",
    response_model=CompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare research papers",
    description=(
        "Retrieve and compare evidence from two or three selected "
        "research papers, then generate a grounded comparison."
    ),
)
def compare_papers(
    request: CompareRequest,
    pipeline: Annotated[
        Any,
        Depends(get_answer_pipeline),
    ],
    inference_lock: Annotated[
        Lock,
        Depends(get_inference_lock),
    ],
) -> CompareResponse:
    """Generate an evidence-grounded comparison of selected papers."""

    request_started = time.perf_counter()

    try:
        # Use the same application-level lock as /ask. The comparison
        # operation shares model instances and GPU resources with research
        # answering, so these operations must not execute concurrently.
        with inference_lock:
            result = pipeline.compare(
                arxiv_ids=request.arxiv_ids,
            )

        # Convert internal evidence records into the public citation schema.
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

        # Measure total server-side processing time, including response
        # construction, but excluding browser and network transfer time.
        total_duration = time.perf_counter() - request_started

        LOGGER.info(
            "Paper comparison completed | duration=%.3fs | "
            "papers=%d | citations=%d | context_tokens=%d | "
            "generated_tokens=%s",
            total_duration,
            len(result.arxiv_ids),
            len(citations),
            result.context_token_count,
            result.generation_token_count,
        )

        return CompareResponse(
            arxiv_ids=result.arxiv_ids,
            answer=result.answer,
            citations=citations,
            context_token_count=result.context_token_count,
            generation_duration_seconds=(
                result.generation_duration_seconds
            ),
            generation_token_count=result.generation_token_count,
            tokens_per_second=result.tokens_per_second,
            total_response_duration_seconds=total_duration,
        )

    except Exception:
        # Avoid logging selected paper content or generated comparison text.
        LOGGER.exception(
            "Paper comparison request failed | elapsed=%.3fs",
            time.perf_counter() - request_started,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The selected papers could not be compared.",
        )
