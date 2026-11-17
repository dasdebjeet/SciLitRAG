
"""
Tests for SciLitRAG API health and readiness endpoints.

Qdrant is mocked so these tests can verify readiness behavior without
requiring a running Docker container or a live vector database.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from scilitrag.api.main import app
from scilitrag.api.routes import health


def test_health_returns_ok() -> None:
    """The liveness endpoint should respond when the API is running."""

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "scilitrag-api",
    }


def test_readiness_returns_ready_when_dependencies_are_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Readiness should succeed when chunks and Qdrant are available."""

    # Create a temporary chunks file to simulate an available corpus.
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text('{"chunk_id": "test"}\n', encoding="utf-8")
    monkeypatch.setattr(health, "DEFAULT_CHUNKS_PATH", chunks_file)

    class FakeQdrantClient:
        """Simulate a healthy Qdrant client."""

        def __init__(self, url: str) -> None:
            self.url = url

        def get_collections(self) -> dict:
            return {"collections": []}

        def close(self) -> None:
            pass

    monkeypatch.setattr(health, "QdrantClient", FakeQdrantClient)

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "qdrant": "connected",
        "chunks": "available",
    }


def test_readiness_fails_when_chunks_file_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Readiness should fail before contacting Qdrant if chunks are missing."""

    missing_file = tmp_path / "missing_chunks.jsonl"
    monkeypatch.setattr(health, "DEFAULT_CHUNKS_PATH", missing_file)

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The processed chunks file is unavailable."
    )
