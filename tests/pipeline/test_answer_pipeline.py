
"""
Unit tests for general-conversation routing in AnswerPipeline.

These tests mock the language model and retrieval components so routing
behaviour can be verified without loading models or using the GPU.
"""

from unittest.mock import MagicMock

import pytest

from scilitrag.generation.generator import GenerationResult
from scilitrag.pipeline import answer_pipeline
from scilitrag.pipeline.context_builder import Evidence


@pytest.mark.parametrize(
    "question",
    [
        "Hello",
        "Hi!",
        "Good morning",
        "Who are you?",
        "Who are you, and what is this chat for?",
        "What is this chat for?",
        "What can you do?",
        "How can you help?",
        "What are your capabilities?",
    ],
)
def test_general_conversation_is_detected(question: str) -> None:
    """Recognized greetings and identity questions should be routed directly."""

    assert answer_pipeline.is_general_conversation(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "What is Recall?",
        "What is Recall in classification and NLP?",
        "How can you help me understand precision and recall?",
        "Explain the proposed research methodology.",
        "What is YOLOv8?",
    ],
)
def test_research_questions_are_not_misclassified(question: str) -> None:
    """Research questions must continue through the normal RAG pipeline."""

    assert answer_pipeline.is_general_conversation(question) is False


def test_general_conversation_skips_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """General questions should generate an answer without initializing retrieval."""

    mock_generator = MagicMock()
    mock_generator.generate.return_value = GenerationResult(
        text="I am SciLitRAG, your research assistant.",
        generation_duration_seconds=1.25,
        generation_token_count=12,
        tokens_per_second=9.6,
    )

    # Replace the real generator so this test does not load Qwen onto the GPU.
    monkeypatch.setattr(
        answer_pipeline,
        "QwenGenerator",
        lambda **kwargs: mock_generator,
    )
    # Retrieval must never be initialized for a general-conversation question.
    def fail_if_retrieval_starts(*args: object, **kwargs: object) -> None:
        pytest.fail("General conversation unexpectedly initialized retrieval.")

    monkeypatch.setattr(
        answer_pipeline,
        "HybridRetriever",
        fail_if_retrieval_starts,
    )

    pipeline = answer_pipeline.AnswerPipeline()

    result = pipeline.answer("Who are you, and what is this chat for?")

    assert result.answer == "I am SciLitRAG, your research assistant."
    assert result.citations == []
    assert result.context_token_count == 0

    mock_generator.generate.assert_called_once()

    # The generator remains alive until the pipeline is explicitly closed.
    mock_generator.close.assert_not_called()

    pipeline.close()

    mock_generator.close.assert_called_once()

    # Confirm the direct-conversation prompt was used.
    generate_kwargs = mock_generator.generate.call_args.kwargs
    assert (
        generate_kwargs["system_prompt"]
        == answer_pipeline.GENERAL_CONVERSATION_SYSTEM_PROMPT
    )
    assert generate_kwargs["max_new_tokens"] == 400


def test_arxiv_filter_prevents_general_conversation_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An explicit arXiv filter must force the normal research retrieval path,
    even when the question resembles a general-conversation question.
    """

    mock_hybrid = MagicMock()
    mock_hybrid.search.return_value = []

    # Replace the retriever with a mock that records its search arguments.
    monkeypatch.setattr(
        answer_pipeline,
        "HybridRetriever",
        lambda **kwargs: mock_hybrid,
    )

    # Generation should not be needed because the mock returns no evidence.
    mock_generator = MagicMock()
    monkeypatch.setattr(
        answer_pipeline,
        "QwenGenerator",
        lambda: mock_generator,
    )

    pipeline = answer_pipeline.AnswerPipeline()

    result = pipeline.answer(
        "Who are you?",
        arxiv_id="2611.17111",
    )

    # The explicit paper filter should reach the retrieval component.
    mock_hybrid.search.assert_called_once()
    assert (
        mock_hybrid.search.call_args.kwargs["arxiv_id"]
        == "2611.17111"
    )

    # No evidence means generation is never initialized or invoked.
    mock_generator.generate.assert_not_called()
    mock_generator.close.assert_not_called()

    # Retrieval resources remain alive until pipeline shutdown.
    mock_hybrid.close.assert_not_called()

    pipeline.close()

    mock_hybrid.close.assert_called_once()
    mock_generator.close.assert_not_called()

    assert result.citations == []
    assert result.context_token_count == 0
    assert "could not find relevant research passages" in result.answer


def test_numeric_source_citation_is_resolved() -> None:
    """A valid numeric citation should resolve to its original source."""

    evidence = Evidence(
        evidence_id="1",
        chunk_id="test-chunk-001",
        arxiv_id="2611.17111v1",
        title="Test Research Paper",
        page_number=2,
        text="A complete source passage.",
        reranker_score=0.95,
    )

    answer, resolved = answer_pipeline.resolve_citations(
        "The paper reports this finding [1].",
        [evidence],
    )

    assert answer == "The paper reports this finding [1]."
    assert len(resolved) == 1
    assert resolved[0].evidence_id == "1"


def test_unknown_numeric_source_citation_is_marked_unverified() -> None:
    """An invented numeric citation must not be accepted as a valid source."""

    answer, resolved = answer_pipeline.resolve_citations(
        "The paper reports this finding [9].",
        [],
    )

    assert answer == "The paper reports this finding [unverified citation: 9]."
    assert resolved == []


def test_build_retrieval_query_includes_recent_conversation() -> None:
    """Follow-up references should be accompanied by recent conversation context."""

    history = [
        {"role": "user", "content": "What method does the paper propose?"},
        {"role": "assistant", "content": "It proposes a transformer-based method."},
    ]

    query = answer_pipeline.build_retrieval_query(
        question="What are its limitations?",
        history=history,
    )

    assert "USER: What method does the paper propose?" in query
    assert "ASSISTANT: It proposes a transformer-based method." in query
    assert "CURRENT RESEARCH QUESTION:" in query
    assert "What are its limitations?" in query


def test_build_retrieval_query_without_history_returns_question() -> None:
    """A first-turn research question should remain unchanged."""

    question = "Explain the proposed research methodology."

    assert answer_pipeline.build_retrieval_query(question) == question


def test_build_user_prompt_separates_history_from_fresh_evidence() -> None:
    """Conversation history must be contextual, not treated as research evidence."""

    assembled_context = MagicMock()
    assembled_context.context = "[1] Fresh evidence from the current retrieval."

    history = [
        {"role": "user", "content": "What method does the paper propose?"},
        {"role": "assistant", "content": "It uses a transformer."},
    ]

    prompt = answer_pipeline.build_user_prompt(
        question="What are its limitations?",
        assembled_context=assembled_context,
        history=history,
    )

    assert "USER: What method does the paper propose?" in prompt
    assert "ASSISTANT: It uses a transformer." in prompt
    assert "What are its limitations?" in prompt
    assert "[1] Fresh evidence from the current retrieval." in prompt
    assert "They are not scientific evidence" in prompt
    assert "Previous assistant responses and their citations are not evidence." in prompt


def test_answer_uses_context_aware_query_for_retrieval_and_reranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Follow-up context should reach both retrieval stages without losing paper scope."""

    mock_hybrid = MagicMock()
    mock_hybrid.search.return_value = [
        {"chunk_id": "test-chunk-001", "text": "A retrieved research passage."}
    ]

    mock_reranker = MagicMock()
    # Returning no ranked candidates lets this test stop before generator loading.
    mock_reranker.rerank.return_value = []

    monkeypatch.setattr(
        answer_pipeline,
        "HybridRetriever",
        lambda **kwargs: mock_hybrid,
    )
    monkeypatch.setattr(
        answer_pipeline,
        "CrossEncoderReranker",
        lambda **kwargs: mock_reranker,
    )

    pipeline = answer_pipeline.AnswerPipeline()

    history = [
        {"role": "user", "content": "What method does this paper propose?"},
        {"role": "assistant", "content": "It proposes a transformer-based method."},
    ]

    question = "What are its limitations?"
    arxiv_id = "2611.17111"

    result = pipeline.answer(
        question=question,
        arxiv_id=arxiv_id,
        history=history,
    )

    expected_query = answer_pipeline.build_retrieval_query(
        question=question,
        history=history,
    )

    # Both retrieval stages must receive the same context-aware query.
    assert mock_hybrid.search.call_args.kwargs["query"] == expected_query
    assert mock_reranker.rerank.call_args.kwargs["query"] == expected_query

    # The selected paper scope must remain active for follow-up questions.
    assert mock_hybrid.search.call_args.kwargs["arxiv_id"] == arxiv_id

    # Persistent retrieval resources must remain alive between requests.
    mock_reranker.close.assert_not_called()
    mock_hybrid.close.assert_not_called()

    pipeline.close()

    mock_reranker.close.assert_called_once()
    mock_hybrid.close.assert_called_once()

    # No ranked evidence means generation should not be initialized.
    assert result.citations == []
    assert result.context_token_count == 0


def test_answer_retrieves_and_reranks_each_selected_paper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Multi-paper follow-ups should search and rerank each selected paper
    independently while preserving the conversation-aware retrieval query.
    """

    arxiv_ids = ["2611.17111", "2611.17112", "2611.17113"]

    mock_hybrid = MagicMock()
    mock_hybrid.search.return_value = [
        {"chunk_id": "test-chunk-001", "text": "A retrieved research passage."}
    ]

    mock_reranker = MagicMock()
    # Empty ranked results stop the pipeline before context assembly and
    # generation, keeping this test focused on retrieval orchestration.
    mock_reranker.rerank.return_value = []

    monkeypatch.setattr(
        answer_pipeline,
        "HybridRetriever",
        lambda **kwargs: mock_hybrid,
    )
    monkeypatch.setattr(
        answer_pipeline,
        "CrossEncoderReranker",
        lambda **kwargs: mock_reranker,
    )

    pipeline = answer_pipeline.AnswerPipeline()

    history = [
        {"role": "user", "content": "Compare the methods used in these papers."},
        {"role": "assistant", "content": "They use different approaches."},
    ]
    question = "Which approach is more computationally efficient?"

    result = pipeline.answer(
        question=question,
        arxiv_ids=arxiv_ids,
        history=history,
    )

    expected_query = answer_pipeline.build_retrieval_query(
        question=question,
        history=history,
    )

    # Each selected paper must receive its own scoped retrieval request.
    assert mock_hybrid.search.call_count == len(arxiv_ids)

    searched_ids = [
        call.kwargs["arxiv_id"]
        for call in mock_hybrid.search.call_args_list
    ]
    assert searched_ids == arxiv_ids

    # Every paper's candidates must be reranked using the same contextual query.
    assert mock_reranker.rerank.call_count == len(arxiv_ids)

    for call in mock_reranker.rerank.call_args_list:
        assert call.kwargs["query"] == expected_query

    # The multi-paper scope must not be reduced to a single-paper filter.
    for call in mock_hybrid.search.call_args_list:
        assert call.kwargs["query"] == expected_query

    # No ranked evidence means generation should not be initialized.
    assert result.citations == []
    assert result.context_token_count == 0
    assert "could not find relevant research passages" in result.answer

    # Retrieval resources remain alive until pipeline shutdown.
    mock_reranker.close.assert_not_called()
    mock_hybrid.close.assert_not_called()

    pipeline.close()

    mock_reranker.close.assert_called_once()
    mock_hybrid.close.assert_called_once()


def test_interleave_candidates_balances_papers_by_rank() -> None:
    """Candidates should be interleaved by rank across paper groups."""

    paper_a = [
        {"chunk_id": "a1", "arxiv_id": "paper-a"},
        {"chunk_id": "a2", "arxiv_id": "paper-a"},
        {"chunk_id": "a3", "arxiv_id": "paper-a"},
    ]
    paper_b = [
        {"chunk_id": "b1", "arxiv_id": "paper-b"},
        {"chunk_id": "b2", "arxiv_id": "paper-b"},
    ]
    paper_c = [
        {"chunk_id": "c1", "arxiv_id": "paper-c"},
    ]

    combined = answer_pipeline.interleave_candidates(
        [paper_a, paper_b, paper_c]
    )

    assert [item["chunk_id"] for item in combined] == [
        "a1", "b1", "c1",
        "a2", "b2",
        "a3",
    ]

    assert [item["arxiv_id"] for item in combined] == [
        "paper-a", "paper-b", "paper-c",
        "paper-a", "paper-b",
        "paper-a",
    ]


def test_interleave_candidates_handles_empty_groups() -> None:
    """Empty paper groups should not disrupt candidate ordering."""

    paper_a = [{"chunk_id": "a1", "arxiv_id": "paper-a"}]
    paper_b: list[dict[str, object]] = []

    combined = answer_pipeline.interleave_candidates(
        [paper_a, paper_b, []]
    )

    assert combined == paper_a


def _make_uploaded_candidate(upload_id: str) -> dict[str, object]:
    """Build a representative uploaded-document chunk for pipeline tests."""

    return {
        "chunk_id": "uploaded-chunk-001",
        "upload_id": upload_id,
        "source_type": "upload",
        "source_id": upload_id,
        "filename": "research-paper.pdf",
        "title": "Uploaded Research Paper",
        "page_number": 1,
        "total_pages": 5,
        "chunk_index": 0,
        "text": "A retrieved passage from the uploaded research paper.",
        "token_count": 12,
        "char_count": 55,
        "arxiv_id": None,
    }


def test_uploaded_document_scope_uses_isolated_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uploaded-document questions must use isolated retrieval, not Qdrant."""

    upload_id = "test-upload-001"
    mock_uploaded = MagicMock()
    mock_uploaded.search.return_value = [
        _make_uploaded_candidate(upload_id)
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = []

    monkeypatch.setattr(
        answer_pipeline,
        "UploadedDocumentRetriever",
        lambda *args, **kwargs: mock_uploaded,
    )
    monkeypatch.setattr(
        answer_pipeline,
        "CrossEncoderReranker",
        lambda **kwargs: mock_reranker,
    )

    def fail_if_hybrid_is_initialized(*args: object, **kwargs: object) -> None:
        pytest.fail("Uploaded-document retrieval initialized HybridRetriever.")

    monkeypatch.setattr(
        answer_pipeline,
        "HybridRetriever",
        fail_if_hybrid_is_initialized,
    )

    pipeline = answer_pipeline.AnswerPipeline()

    # Even a greeting must use the explicitly selected uploaded-document scope.
    result = pipeline.answer(
        question="Hello",
        upload_id=upload_id,
    )

    mock_uploaded.search.assert_called_once()
    assert mock_uploaded.search.call_args.kwargs["query"] == "Hello"
    assert mock_uploaded.search.call_args.kwargs["top_k"] == pipeline.retrieval_k

    mock_reranker.rerank.assert_called_once()
    assert mock_reranker.rerank.call_args.kwargs["query"] == "Hello"

    # Empty reranked results should stop before generation.
    assert result.citations == []
    assert result.context_token_count == 0

    pipeline.close()
    mock_uploaded.close.assert_called_once()
    mock_reranker.close.assert_called_once()


def test_uploaded_document_follow_up_reuses_same_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Follow-up questions should reuse the retriever for the same upload."""

    upload_id = "test-upload-002"
    mock_uploaded = MagicMock()
    mock_uploaded.upload_id = upload_id
    mock_uploaded.search.return_value = [
        _make_uploaded_candidate(upload_id)
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = []

    retriever_factory = MagicMock(return_value=mock_uploaded)

    monkeypatch.setattr(
        answer_pipeline,
        "UploadedDocumentRetriever",
        retriever_factory,
    )
    monkeypatch.setattr(
        answer_pipeline,
        "CrossEncoderReranker",
        lambda **kwargs: mock_reranker,
    )

    pipeline = answer_pipeline.AnswerPipeline()

    history = [
        {"role": "user", "content": "What methodology does this paper use?"},
        {"role": "assistant", "content": "It uses a transformer-based method."},
    ]

    pipeline.answer(
        question="What are its limitations?",
        upload_id=upload_id,
        history=history,
    )

    pipeline.answer(
        question="How does it compare with existing approaches?",
        upload_id=upload_id,
        history=history,
    )

    # The same uploaded-document retriever should serve both requests.
    retriever_factory.assert_called_once()
    assert mock_uploaded.search.call_count == 2

    expected_queries = [
        answer_pipeline.build_retrieval_query(
            "What are its limitations?",
            history,
        ),
        answer_pipeline.build_retrieval_query(
            "How does it compare with existing approaches?",
            history,
        ),
    ]

    actual_queries = [
        call.kwargs["query"]
        for call in mock_uploaded.search.call_args_list
    ]
    assert actual_queries == expected_queries

    pipeline.close()
    mock_uploaded.close.assert_called_once()


def test_uploaded_document_with_no_candidates_returns_without_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty upload retrieval result should not initialize the generator."""

    mock_uploaded = MagicMock()
    mock_uploaded.search.return_value = []

    mock_reranker = MagicMock()

    monkeypatch.setattr(
        answer_pipeline,
        "UploadedDocumentRetriever",
        lambda *args, **kwargs: mock_uploaded,
    )
    monkeypatch.setattr(
        answer_pipeline,
        "CrossEncoderReranker",
        lambda **kwargs: mock_reranker,
    )

    mock_generator = MagicMock()
    monkeypatch.setattr(
        answer_pipeline,
        "QwenGenerator",
        lambda **kwargs: mock_generator,
    )

    pipeline = answer_pipeline.AnswerPipeline()

    result = pipeline.answer(
        question="Summarize the uploaded paper.",
        upload_id="test-upload-003",
    )

    mock_uploaded.search.assert_called_once()
    mock_reranker.rerank.assert_not_called()
    mock_generator.generate.assert_not_called()

    assert result.citations == []
    assert result.context_token_count == 0
    assert result.answer

    pipeline.close()

    mock_uploaded.close.assert_called_once()
    mock_generator.close.assert_not_called()


@pytest.mark.parametrize(
    ("scope_kwargs",),
    [
        ({"arxiv_id": "2611.17111"},),
        ({"arxiv_ids": ["2611.17111", "2611.17112"]},),
    ],
)
def test_uploaded_document_scope_rejects_arxiv_scope(
    scope_kwargs: dict[str, object],
) -> None:
    """An uploaded document and indexed-paper scope cannot be combined."""

    pipeline = answer_pipeline.AnswerPipeline()

    with pytest.raises(ValueError):
        pipeline.answer(
            question="Compare these research findings.",
            upload_id="test-upload-004",
            **scope_kwargs,
        )

    # Invalid scope combinations must fail before resources are initialized.
    assert pipeline.uploaded_retriever is None
    assert pipeline.hybrid is None

    pipeline.close()