
"""
Document-local semantic retrieval for user-uploaded research PDFs.

Unlike DenseRetriever, this component never connects to Qdrant or accesses
the indexed arXiv corpus. It loads one upload, creates page-aware chunks,
embeds them in memory, and searches only those vectors.

The embedding model and tokenizer can be injected during testing to avoid
loading external models or requiring GPU resources.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from scilitrag.core.paths import UPLOADS_DIR

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DEVICE = "cuda"
EXPECTED_DIMENSIONS = 1024
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 80
DEFAULT_TOP_K = 5
DEFAULT_BATCH_SIZE = 32


def _split_into_sentences(text: str) -> list[str]:
    """Split extracted page text at punctuation followed by whitespace."""

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


def _chunk_page_text(
    text: str,
    tokenizer: Any,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int]]:
    """
    Reproduce the corpus chunker's sentence-aware token-window behavior.

    Pages are processed independently so a retrieved chunk always retains
    the page number needed to construct a reliable citation.
    """

    if not text or not text.strip():
        return []

    chunks: list[tuple[str, int]] = []
    current_tokens: list[int] = []

    def emit_chunk() -> None:
        """Emit the current token window and retain its trailing overlap."""

        nonlocal current_tokens

        if not current_tokens:
            return

        chunk_text = tokenizer.decode(
            current_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        if chunk_text:
            chunks.append((chunk_text, len(current_tokens)))

        current_tokens = (
            current_tokens[-overlap:] if overlap > 0 else []
        )

    for sentence in _split_into_sentences(text):
        sentence_tokens = tokenizer.encode(
            sentence,
            add_special_tokens=False,
        )

        remaining_tokens = sentence_tokens

        while remaining_tokens:
            available_capacity = chunk_size - len(current_tokens)

            if available_capacity <= 0:
                emit_chunk()
                available_capacity = chunk_size - len(current_tokens)

            tokens_to_add = remaining_tokens[:available_capacity]
            current_tokens.extend(tokens_to_add)
            remaining_tokens = remaining_tokens[available_capacity:]

            # Long sentences are split at token boundaries without losing
            # content; the next window receives the configured overlap.
            if remaining_tokens:
                emit_chunk()

    if current_tokens:
        final_text = tokenizer.decode(
            current_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        if final_text and (
            not chunks or final_text != chunks[-1][0]
        ):
            chunks.append((final_text, len(current_tokens)))

    return chunks


def _make_chunk_id(
    upload_id: str,
    page_number: int,
    chunk_index: int,
) -> str:
    """Create a stable chunk identifier scoped to one uploaded document."""

    source = f"{upload_id}:page:{page_number}:chunk:{chunk_index}"

    return hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()[:16]


class UploadedDocumentRetriever:
    """
    Retrieve semantically relevant passages from exactly one uploaded PDF.

    The document's vectors remain in process memory and are never written to
    the shared Qdrant collection or the indexed corpus files.

    Args:
        upload_id: Opaque identifier returned by the upload endpoint.
        uploads_dir: Root directory containing uploaded-document folders.
        model_name: SentenceTransformer model used for dense embeddings.
        device: Model execution device.
        chunk_size: Maximum number of content tokens per chunk.
        overlap: Number of trailing tokens carried into the next chunk.
        model: Optional injected embedding model, primarily for tests.
        tokenizer: Optional injected tokenizer, primarily for tests.

    Raises:
        ValueError: If the upload ID, chunk configuration, metadata, or
            generated embeddings are invalid.
        FileNotFoundError: If the requested upload does not exist.
    """

    def __init__(
        self,
        upload_id: str,
        *,
        uploads_dir: Path = UPLOADS_DIR,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("Chunk size must be greater than zero.")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "Overlap must be non-negative and smaller than chunk size."
            )

        if batch_size < 1:
            raise ValueError("Batch size must be greater than zero.")

        try:
            # Parse as a UUID before constructing a path. This prevents a
            # client-supplied upload ID from escaping the uploads directory.
            normalized_upload_id = str(uuid.UUID(upload_id))
        except (ValueError, AttributeError) as exc:
            raise ValueError("Invalid uploaded-document ID.") from exc

        self.upload_id = normalized_upload_id
        self.device = device
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None

        document_dir = (uploads_dir / normalized_upload_id).resolve()
        uploads_root = uploads_dir.resolve()

        if not document_dir.is_relative_to(uploads_root):
            raise ValueError("Upload path is outside the storage directory.")

        metadata_path = document_dir / "metadata.json"
        pages_path = document_dir / "pages.json"

        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        with pages_path.open("r", encoding="utf-8") as pages_file:
            pages = json.load(pages_file)

        if not isinstance(metadata, dict):
            raise TypeError("Uploaded document metadata must be a JSON object.")

        if not isinstance(pages, list):
            raise TypeError("Uploaded page data must be a JSON array.")

        if metadata.get("upload_id") != normalized_upload_id:
            raise ValueError("Upload metadata ID does not match its directory.")

        if not isinstance(pages, list):
            raise TypeError("Uploaded page data must be a JSON array.")

        self.filename = str(metadata.get("filename") or "uploaded-paper.pdf")
        self.title = str(metadata.get("title") or self.filename)

        # Load only the resources that were not injected by the caller.
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.model is None:
            self.model = SentenceTransformer(
                model_name,
                device=device,
            )

        dimensions = self.model.get_embedding_dimension()

        if dimensions != EXPECTED_DIMENSIONS:
            raise ValueError(
                f"Unexpected embedding dimensions: {dimensions}; "
                f"expected {EXPECTED_DIMENSIONS}."
            )

        self._build_chunks(pages, chunk_size, overlap)

        if not self.chunks:
            raise ValueError(
                "The uploaded PDF does not contain any usable text chunks."
            )

        self._embed_chunks(batch_size)

        LOGGER.info(
            "Initialized uploaded-document retrieval | upload_id=%s "
            "| chunks=%d | dimensions=%d | device=%s",
            self.upload_id,
            len(self.chunks),
            EXPECTED_DIMENSIONS,
            device,
        )

    def _build_chunks(
        self,
        pages: list[dict[str, Any]],
        chunk_size: int,
        overlap: int,
    ) -> None:
        """Create page-aware chunk records using the corpus chunking rules."""

        assert self.tokenizer is not None

        for page in pages:
            if not isinstance(page, dict):
                raise TypeError("Every uploaded page must be a JSON object.")

            page_number = page.get("page_number")
            text = page.get("text", "")

            if (
                not isinstance(page_number, int)
                or isinstance(page_number, bool)
                or page_number < 1
            ):
                raise ValueError("Uploaded page has an invalid page number.")

            if not isinstance(text, str):
                raise TypeError("Uploaded page text must be a string.")

            page_chunks = _chunk_page_text(
                text=text,
                tokenizer=self.tokenizer,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            for chunk_index, (chunk_text, token_count) in enumerate(
                page_chunks
            ):
                self.chunks.append(
                    {
                        "chunk_id": _make_chunk_id(
                            self.upload_id,
                            page_number,
                            chunk_index,
                        ),
                        "upload_id": self.upload_id,
                        "source_type": "upload",
                        "source_id": self.upload_id,
                        "filename": self.filename,
                        "title": self.title,
                        "page_number": page_number,
                        "total_pages": page.get("total_pages"),
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "token_count": token_count,
                        "char_count": len(chunk_text),
                        # Uploaded evidence has no arXiv identifier.
                        "arxiv_id": None,
                    }
                )

    def _embed_chunks(self, batch_size: int) -> None:
        """Embed all local chunks using the corpus normalization strategy."""

        assert self.model is not None

        texts = [chunk["text"] for chunk in self.chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        vectors = np.asarray(embeddings, dtype=np.float32)

        expected_shape = (len(self.chunks), EXPECTED_DIMENSIONS)

        if vectors.shape != expected_shape:
            raise ValueError(
                f"Invalid document embedding shape: {vectors.shape}; "
                f"expected {expected_shape}."
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Document embeddings contain non-finite values.")

        norms = np.linalg.norm(vectors, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("Document embeddings are not normalized.")

        self.embeddings = vectors

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Return the highest-scoring passages from this uploaded document.

        Similarity is the dot product of normalized BGE-M3 vectors, equivalent
        to cosine similarity. Scores are ranking signals, not probabilities.
        """

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        if self.model is None or self.embeddings is None:
            raise RuntimeError("Uploaded-document retriever is closed.")

        query_embedding = self.model.encode(
            query.strip(),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        query_vector = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_vector.shape != (EXPECTED_DIMENSIONS,):
            raise ValueError(
                f"Invalid query embedding shape: {query_vector.shape}."
            )

        if not np.isfinite(query_vector).all():
            raise ValueError("Query embedding contains non-finite values.")

        query_norm = float(np.linalg.norm(query_vector))

        if not np.isclose(query_norm, 1.0, atol=1e-3):
            raise ValueError("Query embedding is not normalized.")

        scores = self.embeddings @ query_vector
        result_count = min(top_k, len(self.chunks))

        # Partial selection avoids sorting every chunk when only a few
        # highest-scoring passages are needed.
        candidate_indices = np.argpartition(
            scores,
            -result_count,
        )[-result_count:]

        ranked_indices = candidate_indices[
            np.argsort(scores[candidate_indices])[::-1]
        ]

        results: list[dict[str, Any]] = []

        for index in ranked_indices:
            candidate = dict(self.chunks[int(index)])
            candidate["dense_score"] = float(scores[index])
            results.append(candidate)

        return results

    def close(self) -> None:
        """Release model references and in-memory document vectors."""

        self.embeddings = None
        self.chunks.clear()
        self.model = None
        self.tokenizer = None