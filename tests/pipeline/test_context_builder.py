
"""
Unit tests for citation-aware evidence context assembly.

These tests verify candidate validation, duplicate removal, per-source
limits, token-budget enforcement, and citation metadata for both indexed
arXiv papers and uploaded research documents.
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
    """Build a valid arXiv retrieval candidate."""

    return {
        "chunk_id": chunk_id,
        "arxiv_id": arxiv_id,
        "title": title or f"Research Paper {arxiv_id}",
        "page_number": page_number,
        "text": text or f"Evidence passage for chunk {chunk_id}.",
        "reranker_score": reranker_score,
    }


def make_upload_candidate(
    chunk_id: str,
    upload_id: str,
    *,
    filename: str = "research-paper.pdf",
    title: str = "Uploaded Research Paper",
    text: str = "Evidence from the uploaded research paper.",
    page_number: int = 1,
    reranker_score: float = 0.95,
) -> dict[str, object]:
    """Build an uploaded-document candidate with traceable provenance."""

    return {
        "chunk_id": chunk_id,
        "upload_id": upload_id,
        "source_type": "upload",
        "source_id": upload_id,
        "filename": filename,
        "title": title,
        "page_number": page_number,
        "text": text,
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
        make_candidate("a1", "paper-a", text="A" * 200),
        make_candidate("b1", "paper-b", text="B" * 200),
    ]

    result = ContextBuilder(max_tokens=80).build(candidates)

    assert len(result.evidence) == 1
    assert result.evidence[0].chunk_id == "a1"
    assert result.token_count <= 80
    assert "[1]" in result.context
    assert "[2]" not in result.context


def test_context_builder_returns_empty_context_for_no_candidates() -> None:
    """An empty retrieval result should produce an empty context safely."""

    result = ContextBuilder().build([])

    assert result.context == ""
    assert result.evidence == []
    assert result.token_count == 0


def test_context_builder_preserves_uploaded_document_metadata() -> None:
    """Uploaded evidence must retain its ID, filename, and page number."""

    candidate = make_upload_candidate(
        "upload-chunk-001",
        "12345678-1234-5678-1234-567812345678",
        page_number=7,
    )

    result = ContextBuilder().build([candidate])

    assert len(result.evidence) == 1

    evidence = result.evidence[0]
    assert evidence.arxiv_id is None
    assert evidence.source_type == "upload"
    assert evidence.source_id == "12345678-1234-5678-1234-567812345678"
    assert evidence.filename == "research-paper.pdf"
    assert evidence.page_number == 7

    assert "Uploaded document: research-paper.pdf" in result.context
    assert "Document ID:" in result.context
    assert "Page: 7" in result.context
    assert "arXiv ID:" not in result.context


def test_context_builder_limits_chunks_per_uploaded_document() -> None:
    """The configured source limit must apply to uploaded documents too."""

    upload_id = "12345678-1234-5678-1234-567812345678"
    candidates = [
        make_upload_candidate(f"chunk-{index}", upload_id)
        for index in range(1, 5)
    ]

    result = ContextBuilder(max_chunks_per_paper=3).build(candidates)

    assert len(result.evidence) == 3
    assert all(item.source_id == upload_id for item in result.evidence)


def test_context_builder_rejects_upload_without_filename() -> None:
    """An uploaded citation without a filename must not enter the context."""

    candidate = make_upload_candidate(
        "upload-chunk-001",
        "12345678-1234-5678-1234-567812345678",
    )
    candidate["filename"] = ""

    result = ContextBuilder().build([candidate])

    assert result.evidence == []
    assert result.context == ""


def test_context_builder_preserves_both_source_types() -> None:
    """The builder must support arXiv and upload provenance consistently."""

    upload_id = "12345678-1234-5678-1234-567812345678"
    candidates = [
        make_candidate("arxiv-chunk-001", "2611.17111"),
        make_upload_candidate("upload-chunk-001", upload_id),
    ]

    result = ContextBuilder().build(candidates)

    assert [item.source_type for item in result.evidence] == [
        "arxiv",
        "upload",
    ]
    assert result.evidence[0].arxiv_id == "2611.17111"
    assert result.evidence[1].arxiv_id is None
    assert "[1]" in result.context
    assert "[2]" in result.context