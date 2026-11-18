
"""
API tests for the SciLitRAG paper library endpoint.

These tests use temporary metadata files instead of modifying the
real paper catalogue. This keeps tests isolated, repeatable, and
independent of the actual research corpus.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scilitrag.api.main import app
from scilitrag.core.paths import PAPERS_METADATA_PATH
from scilitrag.api.routes import papers


@pytest.fixture
def test_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """Provide an API client with an isolated metadata catalogue."""

    metadata_path = tmp_path / "papers.json"

    monkeypatch.setattr(
        papers,
        "PAPERS_METADATA_PATH",
        metadata_path,
    )

    return TestClient(app)


def create_valid_paper() -> dict:
    """Return a representative metadata record for API tests."""

    return {
        "arxiv_id": "2611.17111v1",
        "title": "Test Research Paper",
        "authors": ["Author One", "Author Two"],
        "abstract": "A representative research paper abstract.",
        "published": "2026-11-12T17:04:50+00:00",
        "updated": "2026-11-12T17:04:50+00:00",
        "categories": ["cs.LG", "cs.AI"],
        "primary_category": "cs.LG",
        "pdf_url": "https://arxiv.org/pdf/2611.17111v1",
        "pdf_path": "data/raw/2611.17111v1.pdf",
    }


def test_get_papers_returns_validated_records(
    test_client: TestClient,
    tmp_path: Path,
) -> None:
    """A valid catalogue should return papers using the public schema."""

    metadata_path = papers.PAPERS_METADATA_PATH
    metadata_path.write_text(
        json.dumps([create_valid_paper()]),
        encoding="utf-8",
    )

    response = test_client.get("/api/v1/papers")

    assert response.status_code == 200

    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["arxiv_id"] == "2611.17111v1"
    assert payload[0]["title"] == "Test Research Paper"

    # Internal metadata must never leak into the public API.
    assert "pdf_path" not in payload[0]
    assert "updated" not in payload[0]


def test_get_papers_returns_empty_list(
    test_client: TestClient,
) -> None:
    """An empty catalogue should produce a valid empty response."""

    papers.PAPERS_METADATA_PATH.write_text(
        json.dumps([]),
        encoding="utf-8",
    )

    response = test_client.get("/api/v1/papers")

    assert response.status_code == 200
    assert response.json() == []


def test_get_papers_handles_missing_metadata_file(
    test_client: TestClient,
) -> None:
    """A missing catalogue should produce a sanitized HTTP 500."""

    response = test_client.get("/api/v1/papers")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "The paper library could not be loaded."
    )


def test_get_papers_rejects_invalid_records(
    test_client: TestClient,
) -> None:
    """Malformed records should not produce a partial library response."""

    invalid_paper = create_valid_paper()
    del invalid_paper["title"]

    papers.PAPERS_METADATA_PATH.write_text(
        json.dumps([invalid_paper]),
        encoding="utf-8",
    )

    response = test_client.get("/api/v1/papers")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "The paper library could not be loaded."
    )
