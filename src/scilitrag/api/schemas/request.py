
"""
Request schemas for the SciLitRAG API.

These models define and validate the JSON payloads accepted by the
HTTP endpoints. Keeping validation separate from route handlers makes
the API contract explicit and easier to maintain.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ConversationMessage(BaseModel):
    """Represent one prior message supplied as conversational context."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    role: Literal["user", "assistant"] = Field(
        ...,
        description="Whether the message was written by the user or assistant.",
    )

    content: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Text content of the previous conversation message.",
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Reject messages containing only whitespace."""

        if not value.strip():
            raise ValueError("Conversation message cannot be empty.")

        return value


class AskRequest(BaseModel):
    """Validate a research question and its optional conversation context."""

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

    arxiv_ids: list[str] | None = Field(
        default=None,
        min_length=2,
        max_length=3,
        description=(
            "Optional two or three paper IDs used as the retrieval scope "
            "for a multi-paper comparison conversation."
        ),
        examples=[["2611.17101v1", "2611.16871v1"]],
    )

    history: list[ConversationMessage] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "Up to the 12 most recent messages from the active conversation, "
            "excluding the current question."
        ),
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject questions containing only whitespace."""

        if not value.strip():
            raise ValueError("Question cannot contain only whitespace.")

        return value

    @field_validator("arxiv_id")
    @classmethod
    def validate_arxiv_id(cls, value: str | None) -> str | None:
        """Normalize an optional single-paper arXiv identifier."""

        if value is None:
            return None

        value = value.strip()

        if not value:
            return None

        return value

    @field_validator("arxiv_ids")
    @classmethod
    def validate_arxiv_ids(
        cls,
        value: list[str] | None,
    ) -> list[str] | None:
        """Normalize paper identifiers and reject empty or duplicate IDs."""

        if value is None:
            return None

        normalized = [arxiv_id.strip() for arxiv_id in value]

        if any(not arxiv_id for arxiv_id in normalized):
            raise ValueError("arXiv IDs cannot be empty.")

        if len(set(normalized)) != len(normalized):
            raise ValueError("Duplicate arXiv IDs are not allowed.")

        return normalized

    @model_validator(mode="after")
    def validate_retrieval_scope(self) -> AskRequest:
        """Prevent ambiguous requests containing both single- and multi-paper scopes."""

        if self.arxiv_id is not None and self.arxiv_ids is not None:
            raise ValueError(
                "Specify either arxiv_id or arxiv_ids, not both."
            )

        return self


class CompareRequest(BaseModel):
    """Validate a request to compare two or three research papers."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    arxiv_ids: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="Two or three distinct arXiv IDs to compare.",
        examples=[["2611.17101v1", "2611.16871v1"]],
    )

    @field_validator("arxiv_ids")
    @classmethod
    def validate_arxiv_ids(cls, value: list[str]) -> list[str]:
        """Normalize identifiers and reject empty or duplicate selections."""

        normalized = [arxiv_id.strip() for arxiv_id in value]

        if any(not arxiv_id for arxiv_id in normalized):
            raise ValueError("arXiv IDs cannot be empty.")

        if len(set(normalized)) != len(normalized):
            raise ValueError("Duplicate arXiv IDs are not allowed.")

        return normalized
