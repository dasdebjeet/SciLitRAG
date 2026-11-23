"""Tests for isolated semantic retrieval from user-uploaded research PDFs."""

import json
import uuid

import numpy as np
import pytest

from scilitrag.retrieval.uploaded_document_retriever import (
    EXPECTED_DIMENSIONS,
    UploadedDocumentRetriever,
)


class FakeTokenizer:
    """Provide deterministic character-level tokenization for unit tests."""

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """Represent each character as one token without external vocabulary."""
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """Reconstruct text exactly from the character tokens."""
        return "".join(chr(token_id) for token_id in token_ids)


class FakeEmbeddingModel:
    """Generate normalized, deterministic vectors for retrieval tests."""

    def get_embedding_dimension(self) -> int:
        """Match the production BGE-M3 embedding dimension."""
        return EXPECTED_DIMENSIONS

    def encode(
        self,
        sentences,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Map text to simple orthogonal vectors.

        Text containing alpha maps to dimension zero, beta to dimension one,
        and other text to dimension two. This makes ranking deterministic.
        """
        is_single_text = isinstance(sentences, str)
        texts = [sentences] if is_single_text else sentences

        vectors = np.zeros(
            (len(texts), EXPECTED_DIMENSIONS),
            dtype=np.float32,
        )

        for index, text in enumerate(texts):
            lowered_text = text.lower()

            if "alpha" in lowered_text:
                vectors[index, 0] = 1.0
            elif "beta" in lowered_text:
                vectors[index, 1] = 1.0
            else:
                vectors[index, 2] = 1.0

        return vectors[0] if is_single_text else vectors


@pytest.fixture
def upload_id() -> str:
    """Provide a fresh opaque upload identifier for each test."""
    return str(uuid.uuid4())


@pytest.fixture
def uploaded_document(tmp_path, upload_id) -> tuple:
    """Create a realistic two-page upload directory for retrieval tests."""
    document_dir = tmp_path / upload_id
    document_dir.mkdir()

    metadata = {
        "upload_id": upload_id,
        "filename": "research-paper.pdf",
        "title": "Test Research Paper",
        "page_count": 2,
        "total_characters": 75,
    }

    pages = [
        {
            "page_number": 1,
            "total_pages": 2,
            "text": "Alpha research describes the first method.",
        },
        {
            "page_number": 2,
            "total_pages": 2,
            "text": "Beta research describes the second method.",
        },
    ]

    (document_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (document_dir / "pages.json").write_text(
        json.dumps(pages),
        encoding="utf-8",
    )

    return tmp_path, upload_id


def make_retriever(
    uploads_dir,
    upload_id,
    *,
    chunk_size: int = 500,
    overlap: int = 80,
) -> UploadedDocumentRetriever:
    """Construct a retriever without loading production ML resources."""
    return UploadedDocumentRetriever(
        upload_id,
        uploads_dir=uploads_dir,
        chunk_size=chunk_size,
        overlap=overlap,
        model=FakeEmbeddingModel(),
        tokenizer=FakeTokenizer(),
    )


def test_loads_uploaded_document_and_builds_page_aware_chunks(
    uploaded_document,
) -> None:
    """Chunks must retain their upload identity and original page numbers."""
    uploads_dir, upload_id = uploaded_document

    retriever = make_retriever(uploads_dir, upload_id)

    assert retriever.upload_id == upload_id
    assert retriever.filename == "research-paper.pdf"
    assert retriever.title == "Test Research Paper"
    assert len(retriever.chunks) == 2

    first_chunk, second_chunk = retriever.chunks

    assert first_chunk["page_number"] == 1
    assert second_chunk["page_number"] == 2
    assert first_chunk["source_type"] == "upload"
    assert first_chunk["source_id"] == upload_id
    assert first_chunk["upload_id"] == upload_id
    assert first_chunk["arxiv_id"] is None
    assert first_chunk["text"].startswith("Alpha")
    assert second_chunk["text"].startswith("Beta")

    assert first_chunk["chunk_id"] != second_chunk["chunk_id"]
    assert retriever.embeddings.shape == (2, EXPECTED_DIMENSIONS)

    retriever.close()


def test_search_returns_most_relevant_chunks_in_descending_order(
    uploaded_document,
) -> None:
    """Search should rank matching passages above unrelated passages."""
    uploads_dir, upload_id = uploaded_document
    retriever = make_retriever(uploads_dir, upload_id)

    results = retriever.search("alpha research", top_k=2)

    assert len(results) == 2
    assert results[0]["page_number"] == 1
    assert results[0]["dense_score"] > results[1]["dense_score"]
    assert results[0]["dense_score"] == pytest.approx(1.0)
    assert all("dense_score" in result for result in results)

    retriever.close()


def test_search_respects_top_k(uploaded_document) -> None:
    """Search should return no more than the requested number of chunks."""
    uploads_dir, upload_id = uploaded_document
    retriever = make_retriever(uploads_dir, upload_id)

    results = retriever.search("beta research", top_k=1)

    assert len(results) == 1
    assert results[0]["page_number"] == 2

    retriever.close()


def test_search_caps_top_k_at_available_chunk_count(
    uploaded_document,
) -> None:
    """A top_k larger than the document should return all available chunks."""
    uploads_dir, upload_id = uploaded_document
    retriever = make_retriever(uploads_dir, upload_id)

    results = retriever.search("alpha research", top_k=100)

    assert len(results) == len(retriever.chunks)

    retriever.close()


@pytest.mark.parametrize(
    ("query", "top_k", "error_message"),
    [
        ("", 1, "query cannot be empty"),
        ("   ", 1, "query cannot be empty"),
        ("alpha", 0, "top_k must be greater than zero"),
    ],
)
def test_search_rejects_invalid_arguments(
    uploaded_document,
    query: str,
    top_k: int,
    error_message: str,
) -> None:
    """Invalid search inputs must fail before embedding or ranking."""
    uploads_dir, upload_id = uploaded_document
    retriever = make_retriever(uploads_dir, upload_id)

    with pytest.raises(ValueError, match=error_message):
        retriever.search(query, top_k=top_k)

    retriever.close()


def test_rejects_invalid_upload_id(tmp_path) -> None:
    """An arbitrary path-like identifier must not be accepted."""
    with pytest.raises(ValueError, match="Invalid uploaded-document ID"):
        make_retriever(tmp_path, "../../outside")


def test_rejects_metadata_with_mismatched_upload_id(
    uploaded_document,
) -> None:
    """Metadata must belong to the upload directory being requested."""
    uploads_dir, upload_id = uploaded_document
    metadata_path = uploads_dir / upload_id / "metadata.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["upload_id"] = str(uuid.uuid4())
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its directory"):
        make_retriever(uploads_dir, upload_id)


def test_rejects_non_array_page_data(uploaded_document) -> None:
    """The extracted page file must contain a JSON array."""
    uploads_dir, upload_id = uploaded_document
    pages_path = uploads_dir / upload_id / "pages.json"
    pages_path.write_text(json.dumps({"page_number": 1}), encoding="utf-8")

    with pytest.raises(TypeError, match="JSON array"):
        make_retriever(uploads_dir, upload_id)


def test_rejects_non_object_page_entries(uploaded_document) -> None:
    """Each extracted page must be represented by a JSON object."""
    uploads_dir, upload_id = uploaded_document
    pages_path = uploads_dir / upload_id / "pages.json"
    pages_path.write_text(json.dumps(["invalid page"]), encoding="utf-8")

    with pytest.raises(TypeError, match="JSON object"):
        make_retriever(uploads_dir, upload_id)


def test_rejects_invalid_embedding_dimensions(
    uploaded_document,
) -> None:
    """The retriever must enforce the production embedding dimension."""
    uploads_dir, upload_id = uploaded_document

    class InvalidDimensionModel(FakeEmbeddingModel):
        def get_embedding_dimension(self) -> int:
            return 768

    with pytest.raises(ValueError, match="Unexpected embedding dimensions"):
        UploadedDocumentRetriever(
            upload_id,
            uploads_dir=uploads_dir,
            model=InvalidDimensionModel(),
            tokenizer=FakeTokenizer(),
        )


def test_close_releases_resources_and_prevents_future_search(
    uploaded_document,
) -> None:
    """Closing must release vectors and model references."""
    uploads_dir, upload_id = uploaded_document
    retriever = make_retriever(uploads_dir, upload_id)

    retriever.close()

    assert retriever.embeddings is None
    assert retriever.chunks == []
    assert retriever.model is None
    assert retriever.tokenizer is None

    with pytest.raises(RuntimeError, match="retriever is closed"):
        retriever.search("alpha research")
