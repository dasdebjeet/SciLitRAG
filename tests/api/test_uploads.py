
"""API tests for isolated research-paper uploads."""

from collections.abc import Generator
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

import scilitrag.api.routes.uploads as uploads_route
from scilitrag.api.main import app


@pytest.fixture
def test_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """Use temporary upload storage without touching project data."""

    from scilitrag.core import upload_storage

    monkeypatch.setattr(
        upload_storage,
        "UPLOADS_DIR",
        tmp_path / "uploads",
    )

    monkeypatch.setattr(
        uploads_route,
        "store_uploaded_pdf",
        upload_storage.store_uploaded_pdf,
    )

    with TestClient(app) as client:
        yield client


def make_pdf() -> bytes:
    """Generate a valid in-memory PDF for upload tests."""

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "A research paper discussing retrieval augmented generation.",
    )

    content = document.tobytes()
    document.close()

    return content


def test_upload_returns_document_metadata(
    test_client: TestClient,
) -> None:
    """A valid PDF should return its stored document metadata."""

    response = test_client.post(
        "/api/v1/uploads",
        files={
            "file": (
                "research-paper.pdf",
                make_pdf(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201

    payload = response.json()

    assert payload["filename"] == "research-paper.pdf"
    assert payload["page_count"] == 1
    assert payload["total_characters"] > 0
    assert len(payload["upload_id"]) == 36


def test_upload_rejects_non_pdf_filename(
    test_client: TestClient,
) -> None:
    """Files with unsupported extensions should be rejected."""

    response = test_client.post(
        "/api/v1/uploads",
        files={
            "file": (
                "research-paper.txt",
                b"Not a PDF",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415


def test_upload_rejects_invalid_pdf(
    test_client: TestClient,
) -> None:
    """A file with a PDF extension but invalid contents must fail."""

    response = test_client.post(
        "/api/v1/uploads",
        files={
            "file": (
                "broken.pdf",
                b"%PDF-1.7\ninvalid document",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400


def test_upload_rejects_empty_file(
    test_client: TestClient,
) -> None:
    """Empty uploads must not be accepted."""

    response = test_client.post(
        "/api/v1/uploads",
        files={
            "file": (
                "empty.pdf",
                b"",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400


def test_upload_rejects_oversized_file(
    test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint must enforce the configured limit while reading."""

    monkeypatch.setattr(
        uploads_route,
        "MAX_UPLOAD_SIZE_BYTES",
        10,
    )

    response = test_client.post(
        "/api/v1/uploads",
        files={
            "file": (
                "large.pdf",
                make_pdf(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
