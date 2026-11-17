
"""
Response schemas for the SciLitRAG API.

These models convert internal pipeline results into a stable JSON
contract that frontend applications can consume.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CitationResponse(BaseModel):
    """Represent a verified evidence passage and its paper reference."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(
        ...,
        description="Evidence label used in the generated answer.",
        examples=["E1"],
    )

    chunk_id: str = Field(
        ...,
        description="Unique identifier of the source passage.",
    )

    arxiv_id: str = Field(
        ...,
        description="arXiv identifier of the source paper.",
    )

    title: str = Field(
        ...,
        description="Title of the source research paper.",
    )

    page_number: int = Field(
        ...,
        ge=1,
        description="Page number containing the cited passage.",
    )

    text: str = Field(
        ...,
        description="Original evidence passage retrieved from the paper.",
    )

    reranker_score: float = Field(
        ...,
        description="Cross-encoder relevance score; not a probability.",
    )


class AskResponse(BaseModel):
    """Represent the complete result of a research question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        description="Original research question.",
    )

    answer: str = Field(
        ...,
        description="Generated answer grounded in retrieved evidence.",
    )

    citations: list[CitationResponse] = Field(
        default_factory=list,
        description="Verified evidence references used in the answer.",
    )

    context_token_count: int = Field(
        ...,
        ge=0,
        description="Number of tokens used by the assembled evidence context.",
    )
