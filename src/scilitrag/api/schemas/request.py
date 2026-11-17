
"""
Request schemas for the SciLitRAG API.

These models define and validate the JSON payloads accepted by the
HTTP endpoints. Keeping validation separate from route handlers makes
the API contract explicit and easier to maintain.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AskRequest(BaseModel):
    """Validate a research question submitted to the answer endpoint."""

    model_config = ConfigDict(
        # Reject unexpected fields instead of silently ignoring them.
        extra="forbid",
        # Strip surrounding whitespace from string fields.
        str_strip_whitespace=True,
    )

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural-language research question.",
        examples=[
            "How are reversible neural networks used for image enhancement?"
        ],
    )

    arxiv_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Optional arXiv ID to restrict retrieval to one paper.",
        examples=["2611.171"],
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject questions containing only whitespace characters."""

        if not value.strip():
            raise ValueError("Question cannot contain only whitespace.")

        return value

    @field_validator("arxiv_id")
    @classmethod
    def validate_arxiv_id(cls, value: str | None) -> str | None:
        """Normalize an optional arXiv identifier."""

        if value is None:
            return None

        value = value.strip()

        if not value:
            return None

        return value
