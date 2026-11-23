
"""
Evidence context assembly for SciLitRAG.

Converts reranked retrieval candidates into a compact, structured context
for the language model while preserving traceable source metadata.

Responsibilities:
    - Validate candidate passages and source-specific citation metadata.
    - Remove duplicate chunks.
    - Limit chunks contributed by a single source.
    - Enforce a configurable context token budget.
    - Assign stable evidence labels within the assembled context.
    - Return formatted context and citation metadata.

The generator receives numeric source labels such as [1] and [2].
The application resolves those labels into source metadata and page
numbers after generation. Sources may be indexed arXiv papers or
user-uploaded research documents.
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
    """A validated passage and its traceable source metadata."""

    evidence_id: str
    chunk_id: str
    arxiv_id: str | None
    title: str
    page_number: int
    text: str
    reranker_score: float
    source_type: str = "arxiv"
    source_id: str | None = None
    filename: str | None = None


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
        # Retain the parameter name for compatibility with existing callers.
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

        # Conservative fallback estimate, not a tokenizer equivalent.
        return max(1, (len(text) + 3) // 4)

    @staticmethod
    def _validate_candidate(candidate: dict[str, Any]) -> bool:
        """Validate passage fields and metadata for the declared source."""

        for field in ("chunk_id", "title", "text"):
            value = candidate.get(field)

            if not isinstance(value, str) or not value.strip():
                LOGGER.warning(
                    "Skipping candidate with invalid field '%s'.",
                    field,
                )
                return False

        source_type = candidate.get("source_type", "arxiv")

        if source_type == "arxiv":
            source_id = candidate.get("source_id") or candidate.get(
                "arxiv_id"
            )
            if not isinstance(source_id, str) or not source_id.strip():
                LOGGER.warning("Skipping candidate with invalid arXiv ID.")
                return False

        elif source_type == "upload":
            source_id = candidate.get("source_id") or candidate.get(
                "upload_id"
            )
            filename = candidate.get("filename")

            if not isinstance(source_id, str) or not source_id.strip():
                LOGGER.warning("Skipping candidate with invalid upload ID.")
                return False

            if not isinstance(filename, str) or not filename.strip():
                LOGGER.warning("Skipping uploaded candidate with invalid filename.")
                return False

        else:
            LOGGER.warning(
                "Skipping candidate with unsupported source type '%s'.",
                source_type,
            )
            return False

        page_number = candidate.get("page_number")
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            LOGGER.warning("Skipping candidate with invalid page number.")
            return False

        return True

    @staticmethod
    def _source_metadata(
        candidate: dict[str, Any],
    ) -> tuple[str, str, str | None, str | None]:
        """Normalize source metadata while retaining legacy arXiv support."""

        source_type = candidate.get("source_type", "arxiv")

        if source_type == "upload":
            source_id = candidate.get("source_id") or candidate["upload_id"]
            return source_type, source_id, None, candidate["filename"]

        arxiv_id = candidate.get("source_id") or candidate["arxiv_id"]
        return source_type, arxiv_id, arxiv_id, None

    @staticmethod
    def _format_section(evidence: Evidence) -> str:
        """Format a passage with the appropriate source identity."""

        if evidence.source_type == "upload":
            source_details = (
                f"Uploaded document: {evidence.filename}\n"
                f"Document ID: {evidence.source_id}\n"
            )
        else:
            source_details = f"arXiv ID: {evidence.arxiv_id}\n"

        return (
            f"[{evidence.evidence_id}]\n"
            f"Paper: {evidence.title}\n"
            f"{source_details}"
            f"Page: {evidence.page_number}\n"
            f"Passage:\n{evidence.text}\n"
        )

    def build(
        self,
        candidates: list[dict[str, Any]],
    ) -> AssembledContext:
        """Assemble reranked candidates into a citation-aware context."""

        if not candidates:
            return AssembledContext(context="", evidence=[], token_count=0)

        selected: list[Evidence] = []
        seen_chunks: set[str] = set()
        source_counts: dict[tuple[str, str], int] = defaultdict(int)

        # Preserve reranker ordering so higher-ranked passages are preferred.
        for candidate in candidates:
            if not self._validate_candidate(candidate):
                continue

            chunk_id = candidate["chunk_id"]
            if chunk_id in seen_chunks:
                continue

            source_type, source_id, arxiv_id, filename = (
                self._source_metadata(candidate)
            )
            source_key = (source_type, source_id)

            if source_counts[source_key] >= self.max_chunks_per_paper:
                continue

            selected.append(
                Evidence(
                    evidence_id=str(len(selected) + 1),
                    chunk_id=chunk_id,
                    arxiv_id=arxiv_id,
                    title=candidate["title"],
                    page_number=candidate["page_number"],
                    text=candidate["text"].strip(),
                    reranker_score=float(
                        candidate.get("reranker_score", 0.0)
                    ),
                    source_type=source_type,
                    source_id=source_id,
                    filename=filename,
                )
            )
            seen_chunks.add(chunk_id)
            source_counts[source_key] += 1

        # Enforce the context budget using the same headers sent to the model.
        included: list[Evidence] = []
        total_tokens = 0

        for evidence in selected:
            section_tokens = self._count_tokens(
                self._format_section(evidence)
            )

            if total_tokens + section_tokens > self.max_tokens:
                LOGGER.info(
                    "Context budget reached; stopping at %d passages.",
                    len(included),
                )
                break

            included.append(evidence)
            total_tokens += section_tokens

        # Reassign labels after budget selection. No excluded passage can
        # accidentally remain addressable through a numeric citation.
        final_evidence = [
            Evidence(
                evidence_id=str(index),
                chunk_id=item.chunk_id,
                arxiv_id=item.arxiv_id,
                title=item.title,
                page_number=item.page_number,
                text=item.text,
                reranker_score=item.reranker_score,
                source_type=item.source_type,
                source_id=item.source_id,
                filename=item.filename,
            )
            for index, item in enumerate(included, start=1)
        ]

        final_context = "\n".join(
            self._format_section(item) for item in final_evidence
        )

        return AssembledContext(
            context=final_context,
            evidence=final_evidence,
            token_count=self._count_tokens(final_context),
        )
