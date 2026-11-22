
"""
Unit tests for citation-aware evidence context assembly.

These tests verify candidate validation, duplicate removal, per-paper
limits, token-budget enforcement, and stable citation metadata without
loading the language model or embedding components.
"""

from scilitrag.pipeline.context_builder import ContextBuilder


def make_candidate(
    chunk_id: str,
    arxiv_id: str,
    *,
    title: str | None = None,
    text: str | None = None,
    page_number: int = 1,
    reranker_score: float = 0.95,
) -> dict[str, object]:
    """Build a valid retrieval candidate with predictable source metadata."""

    return {
        "chunk_id": chunk_id,
        "arxiv_id": arxiv_id,
        "title": title or f"Research Paper {arxiv_id}",
        "page_number": page_number,
        "text": text or f"Evidence passage for chunk {chunk_id}.",
        "reranker_score": reranker_score,
    }


def test_context_builder_preserves_evidence_from_multiple_papers() -> None:
    """The assembled context should retain evidence from different papers."""

    candidates = [
        make_candidate("a1", "paper-a"),
        make_candidate("b1", "paper-b"),
        make_candidate("c1", "paper-c"),
    ]

    result = ContextBuilder().build(candidates)

    assert [item.arxiv_id for item in result.evidence] == [
        "paper-a",
        "paper-b",
        "paper-c",
    ]
    assert [item.evidence_id for item in result.evidence] == ["1", "2", "3"]

    for evidence_id in ("[1]", "[2]", "[3]"):
        assert evidence_id in result.context


def test_context_builder_limits_chunks_per_paper() -> None:
    """No paper should contribute more than the configured chunk limit."""

    candidates = [
        make_candidate(f"a{index}", "paper-a")
        for index in range(1, 5)
    ] + [
        make_candidate(f"b{index}", "paper-b")
        for index in range(1, 3)
    ]

    result = ContextBuilder(max_chunks_per_paper=3).build(candidates)

    paper_a_count = sum(
        item.arxiv_id == "paper-a"
        for item in result.evidence
    )
    paper_b_count = sum(
        item.arxiv_id == "paper-b"
        for item in result.evidence
    )

    assert paper_a_count == 3
    assert paper_b_count == 2
    assert len(result.evidence) == 5


def test_context_builder_removes_duplicate_chunks() -> None:
    """A chunk must not appear more than once in the assembled context."""

    candidates = [
        make_candidate("duplicate-001", "paper-a"),
        make_candidate("unique-002", "paper-b"),
        make_candidate("duplicate-001", "paper-a"),
    ]

    result = ContextBuilder().build(candidates)

    chunk_ids = [item.chunk_id for item in result.evidence]

    assert chunk_ids == ["duplicate-001", "unique-002"]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_context_builder_enforces_token_budget() -> None:
    """Evidence exceeding the context budget must not be included."""

    candidates = [
        make_candidate(
            "a1",
            "paper-a",
            text="A" * 200,
        ),
        make_candidate(
            "b1",
            "paper-b",
            text="B" * 200,
        ),
    ]

    # The fallback tokenizer estimates approximately four characters per
    # token. This budget allows the first passage but not both passages.
    result = ContextBuilder(max_tokens=80).build(candidates)

    assert len(result.evidence) == 1
    assert result.evidence[0].chunk_id == "a1"
    assert result.token_count <= 80

    # Only evidence actually included in the context may be cited.
    assert "[1]" in result.context
    assert "[2]" not in result.context


def test_context_builder_returns_empty_context_for_no_candidates() -> None:
    """An empty retrieval result should produce an empty context safely."""

    result = ContextBuilder().build([])

    assert result.context == ""
    assert result.evidence == []
    assert result.token_count == 0
