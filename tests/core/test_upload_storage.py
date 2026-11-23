
"""Tests for isolated PDF upload storage and extraction."""

import json

import pymupdf
import pytest

from scilitrag.core.upload_storage import store_uploaded_pdf


def make_pdf(*pages: str) -> bytes:
    """Generate a small in-memory PDF for deterministic storage tests."""

    document = pymupdf.open()

    for page_text in pages:
        page = document.new_page()
        page.insert_text((72, 72), page_text)

    content = document.tobytes()
    document.close()

    return content


def test_stores_pdf_and_extracted_pages(tmp_path) -> None:
    """A valid upload should produce isolated files and page metadata."""

    content = make_pdf(
        "First page: research methodology.",
        "Second page: experimental results.",
    )

    result = store_uploaded_pdf(
        content,
        "research-paper.pdf",
        uploads_dir=tmp_path,
    )

    document_dir = tmp_path / result.upload_id

    assert (document_dir / "document.pdf").read_bytes() == content
    assert result.filename == "research-paper.pdf"
    assert result.page_count == 2
    assert result.total_characters > 0

    pages = json.loads((document_dir / "pages.json").read_text())
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2


def test_rejects_non_pdf_content(tmp_path) -> None:
    """Non-PDF content must be rejected without creating an upload."""

    with pytest.raises(ValueError, match="valid PDF"):
        store_uploaded_pdf(
            b"This is not a PDF.",
            "paper.pdf",
            uploads_dir=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_rejects_oversized_pdf(tmp_path) -> None:
    """The configured byte limit must be enforced before PDF parsing."""

    with pytest.raises(ValueError, match="maximum file size"):
        store_uploaded_pdf(
            make_pdf("Research content."),
            "paper.pdf",
            uploads_dir=tmp_path,
            max_size_bytes=10,
        )

    assert list(tmp_path.iterdir()) == []


def test_rejects_excessive_page_count(tmp_path) -> None:
    """The configured page limit must be enforced during extraction."""

    with pytest.raises(ValueError, match="maximum of 1 pages"):
        store_uploaded_pdf(
            make_pdf("Page one.", "Page two."),
            "paper.pdf",
            uploads_dir=tmp_path,
            max_pages=1,
        )

    assert list(tmp_path.iterdir()) == []


def test_rejects_non_pdf_filename(tmp_path) -> None:
    """Uploads must use a PDF filename even when their contents are PDFs."""

    with pytest.raises(ValueError, match="Only PDF"):
        store_uploaded_pdf(
            make_pdf("Research content."),
            "paper.txt",
            uploads_dir=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_sanitizes_filename_and_uses_opaque_storage_id(tmp_path) -> None:
    """Client filename paths must never influence the storage directory."""

    result = store_uploaded_pdf(
        make_pdf("Research content."),
        "../../private/research.pdf",
        uploads_dir=tmp_path,
    )

    assert result.filename == "research.pdf"
    assert (tmp_path / result.upload_id).is_dir()
    assert len(list(tmp_path.iterdir())) == 1
