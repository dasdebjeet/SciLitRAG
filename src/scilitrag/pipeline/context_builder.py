
"""
Evidence context assembly for SciLitRAG.

Converts reranked retrieval candidates into a compact, structured context
for the language model while preserving traceable source metadata.

Responsibilities:
    - Validate candidate passages and citation metadata.
    - Remove duplicate chunks.
    - Optionally limit chunks contributed by a single paper.
    - Enforce a configurable context token budget.
    - Assign stable evidence labels within the assembled context.
    - Return both formatted context and citation metadata.

Design:
    The generator receives numeric source labels such as [1] and [2].
    The application resolves those labels into paper titles, arXiv IDs,
    and page numbers after generation.

Limitations:
    - Token budgeting uses the supplied tokenizer when available.
    - Without a tokenizer, a conservative character-based estimate is used.
    - Evidence labels are stable only within one assembled answer.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol


LOGGER = logging.getLogger(__name__)


class TokenizerProtocol(Protocol):
    """Minimal tokenizer interface required by the context builder."""

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """Return token IDs for the supplied text."""
        ...


@dataclass(frozen=True)
class Evidence:
    """A validated passage and its source metadata."""

    evidence_id: str
    chunk_id: str
    arxiv_id: str
    title: str
    page_number: int
    text: str
    reranker_score: float


@dataclass(frozen=True)
class AssembledContext:
    """Formatted model context and its corresponding evidence records."""

    context: str
    evidence: list[Evidence]
    token_count: int


class ContextBuilder:
    """Build a token-bounded, citation-aware context from reranked chunks."""

    def __init__(
        self,
        tokenizer: TokenizerProtocol | None = None,
        max_tokens: int = 6000,
        max_chunks_per_paper: int = 3,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than zero.")

        if max_chunks_per_paper < 1:
            raise ValueError(
                "max_chunks_per_paper must be greater than zero."
            )

        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_chunks_per_paper = max_chunks_per_paper

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the generator tokenizer when available."""

        if self.tokenizer is not None:
            return len(
                self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )

        # Fallback estimate: approximately four characters per token.
        # This is deliberately approximate, not a tokenizer equivalent.
        return max(1, (len(text) + 3) // 4)

    @staticmethod
    def _validate_candidate(
        candidate: dict[str, Any],
    ) -> bool:
        """Check whether a candidate contains the required citation data."""

        required_strings = (
            "chunk_id",
            "arxiv_id",
            "title",
            "text",
        )

        for field in required_strings:
            value = candidate.get(field)

            if not isinstance(value, str) or not value.strip():
                LOGGER.warning(
                    "Skipping candidate with invalid field '%s'.",
                    field,
                )
                return False

        page_number = candidate.get("page_number")

        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            LOGGER.warning(
                "Skipping candidate with invalid page number."
            )
            return False

        return True

    def build(
        self,
        candidates: list[dict[str, Any]],
    ) -> AssembledContext:
        """
        Assemble reranked candidates into a citation-aware context.

        Candidates are expected to be ordered by descending reranker
        relevance, as returned by CrossEncoderReranker.rerank().
        """

        if not candidates:
            return AssembledContext(
                context="",
                evidence=[],
                token_count=0,
            )

        selected: list[Evidence] = []
        seen_chunks: set[str] = set()
        paper_counts: dict[str, int] = defaultdict(int)

        # Process in reranker order so higher-ranked passages are preferred.
        for candidate in candidates:

            if not self._validate_candidate(candidate):
                continue

            chunk_id = candidate["chunk_id"]
            arxiv_id = candidate["arxiv_id"]

            # Never include the same chunk more than once.
            if chunk_id in seen_chunks:
                continue

            # Prevent one paper from dominating the assembled context.
            if paper_counts[arxiv_id] >= self.max_chunks_per_paper:
                continue

            evidence_id = str(len(selected) + 1)

            evidence = Evidence(
                evidence_id=evidence_id,
                chunk_id=chunk_id,
                arxiv_id=arxiv_id,
                title=candidate["title"],
                page_number=candidate["page_number"],
                text=candidate["text"].strip(),
                reranker_score=float(
                    candidate.get("reranker_score", 0.0)
                ),
            )

            selected.append(evidence)
            seen_chunks.add(chunk_id)
            paper_counts[arxiv_id] += 1

        # Build the context incrementally, accounting for evidence headers.
        included: list[Evidence] = []
        context_parts: list[str] = []
        total_tokens = 0

        for evidence in selected:
            section = (
                f"[{evidence.evidence_id}]\n"
                f"Paper: {evidence.title}\n"
                f"arXiv ID: {evidence.arxiv_id}\n"
                f"Page: {evidence.page_number}\n"
                f"Passage:\n{evidence.text}\n"
            )

            section_tokens = self._count_tokens(section)

            if total_tokens + section_tokens > self.max_tokens:
                LOGGER.info(
                    "Context budget reached; stopping at %d passages.",
                    len(included),
                )
                break

            context_parts.append(section)
            included.append(evidence)
            total_tokens += section_tokens

        # Return only evidence that actually made it into the context.
        # This prevents citations to passages the model never received.
        final_evidence = [
            Evidence(
                evidence_id=str(index),
                chunk_id=item.chunk_id,
                arxiv_id=item.arxiv_id,
                title=item.title,
                page_number=item.page_number,
                text=item.text,
                reranker_score=item.reranker_score,
            )
            for index, item in enumerate(included, start=1)
        ]

        # Rebuild the context if labels changed after budget selection.
        final_parts: list[str] = []

        for item in final_evidence:
            final_parts.append(
                f"[{item.evidence_id}]\n"
                f"Paper: {item.title}\n"
                f"arXiv ID: {item.arxiv_id}\n"
                f"Page: {item.page_number}\n"
                f"Passage:\n{item.text}\n"
            )

        final_context = "\n".join(final_parts)

        return AssembledContext(
            context=final_context,
            evidence=final_evidence,
            token_count=self._count_tokens(final_context),
        )
