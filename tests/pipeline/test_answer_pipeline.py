
"""
Unit tests for general-conversation routing in AnswerPipeline.

These tests mock the language model and retrieval components so routing
behaviour can be verified without loading models or using the GPU.
"""

from unittest.mock import MagicMock

import pytest

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
    mock_generator.generate.return_value = "I am SciLitRAG, your research assistant."

    # Replace the real generator so this test does not load Qwen onto the GPU.
    monkeypatch.setattr(
        answer_pipeline,
        "QwenGenerator",
        lambda: mock_generator,
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

    # Retrieval resources must be released even when no evidence is found.
    mock_hybrid.close.assert_called_once()

    # No evidence means no generator initialization or generation.
    mock_generator.generate.assert_not_called()
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