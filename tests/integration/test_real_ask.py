
"""Integration tests for the real SciLitRAG API pipeline.

These tests exercise the actual retrieval, reranking, and generation
components. They are opt-in because they require Qdrant and GPU resources.
"""

import os

import pytest
from fastapi.testclient import TestClient

from scilitrag.api.main import app


@pytest.mark.integration
def test_real_rag_pipeline_returns_grounded_answer() -> None:
    """Verify that the real RAG pipeline returns an answer with citations."""

    # Prevent accidental GPU inference during ordinary test runs.
    if os.getenv("RUN_REAL_RAG_TEST") != "1":
        pytest.skip(
            "Set RUN_REAL_RAG_TEST=1 to execute real GPU inference."
        )

    # Use the actual FastAPI application without starting a Uvicorn server.
    client = TestClient(app)

    # Submit a research question through the real API endpoint.
    response = client.post(
        "/api/v1/ask",
        json={
            "question": (
                "How does the study quantify pore-space evolution "
                "and fluid displacement?"
            ),
            "arxiv_id": "2611.17111v1",
        },
    )

    # Verify that the API successfully processed the request.
    assert response.status_code == 200

    payload = response.json()

    # Verify the generated answer and context information.
    assert payload["question"]
    assert payload["answer"].strip()
    assert payload["context_token_count"] > 0

    # A grounded answer should contain supporting citations.
    assert len(payload["citations"]) > 0

    # Validate the structure and essential fields of every citation.
    for citation in payload["citations"]:
        assert citation["evidence_id"]
        assert citation["chunk_id"]
        assert citation["arxiv_id"]
        assert citation["title"]
        assert citation["page_number"] >= 1
        assert citation["text"].strip()
