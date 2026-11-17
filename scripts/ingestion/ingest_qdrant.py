
"""Validate and ingest SciLitRAG chunk embeddings into Qdrant.

The script maps each source chunk to one Qdrant point containing a named
dense vector and its original text/metadata. It performs a full validation
pass before uploading, uses deterministic UUIDs for idempotent upserts, and
streams embedding records in batches to keep memory usage predictable.

Expected inputs:
    - data/processed/chunks.jsonl
    - data/processed/embeddings.jsonl

Expected Qdrant collection:
    - Name: scilit_rag
    - Named vector: dense
    - Dimensions: 1024
    - Distance: Cosine

The script does not create, delete, or recreate collections. Run
scripts/utilities/setup_qdrant.py first if the collection has not been initialized.
"""

from __future__ import annotations
from scilitrag.core.paths import PROCESSED_DATA_DIR

import argparse
import hashlib
import json
import logging
import math
import uuid
from pathlib import Path
from typing import Any, Iterator

from qdrant_client import QdrantClient, models


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------



DEFAULT_CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"
DEFAULT_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "embeddings.jsonl"

QDRANT_URL = "http://localhost:16333"
COLLECTION_NAME = "scilit_rag"
VECTOR_NAME = "dense"

EXPECTED_DIMENSIONS = 1024
EXPECTED_MODEL = "BAAI/bge-m3"
EXPECTED_DISTANCE = models.Distance.COSINE

DEFAULT_BATCH_SIZE = 128
NORMALIZATION_TOLERANCE = 1e-3

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure consistent timestamped logging for the ingestion pipeline."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# JSONL Utilities
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file with line-specific error messages.

    Args:
        path: JSONL file to read.

    Yields:
        One decoded JSON object per non-empty line.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If a line contains invalid JSON or is not an object.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, line {line_number}."
                )

            yield record


def load_chunks(path: Path) -> dict[str, dict[str, Any]]:
    """Load chunk records into a lookup indexed by their deterministic IDs.

    The chunk file is comparatively small, so keeping its text and metadata
    in memory allows the embedding file to be validated and uploaded as a
    stream without retaining all 10,092 embedding vectors.

    Args:
        path: Source chunk JSONL file.

    Returns:
        Mapping from chunk_id to its complete source record.

    Raises:
        ValueError: If a chunk is missing required fields or has a duplicate ID.
    """

    chunks: dict[str, dict[str, Any]] = {}

    for record in read_jsonl(path):
        chunk_id = record.get("chunk_id")
        text = record.get("text")

        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("A chunk record has a missing or invalid chunk_id.")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {chunk_id} has missing or empty text.")

        if chunk_id in chunks:
            raise ValueError(f"Duplicate chunk_id in chunk file: {chunk_id}")

        required_fields = (
            "arxiv_id",
            "title",
            "page_number",
            "total_pages",
            "chunk_index",
            "token_count",
            "char_count",
            "pdf_path",
        )

        missing_fields = [
            field for field in required_fields if field not in record
        ]

        if missing_fields:
            raise ValueError(
                f"Chunk {chunk_id} is missing fields: {missing_fields}"
            )

        chunks[chunk_id] = record

    if not chunks:
        raise ValueError(f"No chunk records found in {path}.")

    LOGGER.info("Loaded %d source chunks.", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Embedding Validation
# ---------------------------------------------------------------------------

def calculate_text_hash(text: str) -> str:
    """Return the SHA-256 hash of the exact UTF-8 encoded source text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_embedding_record(
    record: dict[str, Any],
    chunks: dict[str, dict[str, Any]],
) -> str:
    """Validate one embedding and its relationship to the original chunk.

    Args:
        record: Embedding JSON object.
        chunks: Source chunks indexed by chunk_id.

    Returns:
        The validated chunk_id.

    Raises:
        ValueError: If the record is missing, stale, malformed, or inconsistent.
    """

    chunk_id = record.get("chunk_id")

    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("An embedding record has a missing or invalid chunk_id.")

    if chunk_id not in chunks:
        raise ValueError(f"Embedding references an unknown chunk: {chunk_id}")

    chunk = chunks[chunk_id]

    expected_hash = calculate_text_hash(chunk["text"])
    actual_hash = record.get("text_hash")

    if actual_hash != expected_hash:
        raise ValueError(
            f"Text hash mismatch for chunk {chunk_id}. "
            "The embedding may be stale or associated with different text."
        )

    if record.get("model") != EXPECTED_MODEL:
        raise ValueError(
            f"Unexpected embedding model for {chunk_id}: "
            f"{record.get('model')!r}; expected {EXPECTED_MODEL!r}."
        )

    if record.get("dimensions") != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Unexpected dimensions for {chunk_id}: "
            f"{record.get('dimensions')!r}; expected {EXPECTED_DIMENSIONS}."
        )

    if record.get("normalized") is not True:
        raise ValueError(f"Embedding for {chunk_id} is not marked as normalized.")

    vector = record.get("embedding")

    if not isinstance(vector, list) or len(vector) != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Invalid vector length for {chunk_id}; "
            f"expected {EXPECTED_DIMENSIONS} values."
        )

    squared_norm = 0.0

    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Non-numeric vector value in chunk {chunk_id}.")

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise ValueError(f"Non-finite vector value in chunk {chunk_id}.")

        squared_norm += numeric_value * numeric_value

    vector_norm = math.sqrt(squared_norm)

    if abs(vector_norm - 1.0) > NORMALIZATION_TOLERANCE:
        raise ValueError(
            f"Embedding for {chunk_id} is not L2-normalized "
            f"(norm={vector_norm:.6f})."
        )

    return chunk_id


def validate_all_embeddings(
    path: Path,
    chunks: dict[str, dict[str, Any]],
) -> int:
    """Validate the complete embedding file before any Qdrant writes.

    This pass detects duplicate IDs, missing embeddings, unknown chunks,
    stale text hashes, incorrect dimensions, and invalid vector values.

    Args:
        path: Embedding JSONL file.
        chunks: Source chunks indexed by chunk_id.

    Returns:
        Number of validated embedding records.

    Raises:
        ValueError: If the embedding file does not match the chunk dataset.
    """

    seen_ids: set[str] = set()
    validated_count = 0

    LOGGER.info("Starting full embedding integrity validation.")

    for record in read_jsonl(path):
        chunk_id = validate_embedding_record(record, chunks)

        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate embedding record: {chunk_id}")

        seen_ids.add(chunk_id)
        validated_count += 1

        if validated_count % 1000 == 0:
            LOGGER.info(
                "Validated %d / %d embedding records.",
                validated_count,
                len(chunks),
            )

    missing_ids = set(chunks) - seen_ids

    if missing_ids:
        examples = sorted(missing_ids)[:5]
        raise ValueError(
            f"{len(missing_ids)} chunks have no embedding. "
            f"Example missing IDs: {examples}"
        )

    if validated_count != len(chunks):
        raise ValueError(
            f"Record count mismatch: {validated_count} embeddings "
            f"for {len(chunks)} chunks."
        )

    LOGGER.info(
        "Embedding validation passed: %d records, %d dimensions.",
        validated_count,
        EXPECTED_DIMENSIONS,
    )

    return validated_count


# ---------------------------------------------------------------------------
# Qdrant Collection Verification
# ---------------------------------------------------------------------------

def verify_collection(client: QdrantClient) -> None:
    """Verify that the existing collection has the required vector settings.

    The function deliberately does not create or recreate collections.
    This prevents accidental loss of previously ingested data.

    Args:
        client: Connected Qdrant client.

    Raises:
        ValueError: If the collection is missing or configured incorrectly.
    """

    if not client.collection_exists(COLLECTION_NAME):
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' does not exist. "
            "Run scripts/utilities/setup_qdrant.py first."
        )

    collection = client.get_collection(COLLECTION_NAME)
    vector_config = collection.config.params.vectors

    if not isinstance(vector_config, dict):
        raise ValueError("Expected a collection configured with named vectors.")

    if VECTOR_NAME not in vector_config:
        raise ValueError(f"Collection is missing named vector '{VECTOR_NAME}'.")

    actual_config = vector_config[VECTOR_NAME]

    if actual_config.size != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Collection dimension mismatch: expected {EXPECTED_DIMENSIONS}, "
            f"found {actual_config.size}."
        )

    if actual_config.distance != EXPECTED_DISTANCE:
        raise ValueError(
            f"Collection distance mismatch: expected {EXPECTED_DISTANCE}, "
            f"found {actual_config.distance}."
        )

    LOGGER.info(
        "Collection verified: %s | vector=%s | dimensions=%d | distance=%s",
        COLLECTION_NAME,
        VECTOR_NAME,
        actual_config.size,
        actual_config.distance,
    )


# ---------------------------------------------------------------------------
# Point Construction and Upload
# ---------------------------------------------------------------------------

def make_point_id(chunk_id: str) -> str:
    """Generate a stable UUID for a chunk using UUID version 5.

    Qdrant accepts UUID point IDs. The original chunk_id is retained in the
    payload so that application code can still refer to the source record.
    """

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"scilitrag:{chunk_id}"))


def build_point(
    embedding_record: dict[str, Any],
    chunk: dict[str, Any],
) -> models.PointStruct:
    """Construct a Qdrant point from one embedding and its source chunk.

    Args:
        embedding_record: Validated embedding record.
        chunk: Corresponding source chunk and metadata.

    Returns:
        A Qdrant PointStruct with a named dense vector and searchable payload.
    """

    chunk_id = chunk["chunk_id"]

    payload = {
        "chunk_id": chunk_id,
        "text_hash": embedding_record["text_hash"],
        "embedding_model": embedding_record["model"],
        "embedding_dimensions": embedding_record["dimensions"],
        "arxiv_id": chunk["arxiv_id"],
        "title": chunk["title"],
        "page_number": chunk["page_number"],
        "total_pages": chunk["total_pages"],
        "chunk_index": chunk["chunk_index"],
        "text": chunk["text"],
        "token_count": chunk["token_count"],
        "char_count": chunk["char_count"],
        "pdf_path": chunk["pdf_path"],
    }

    return models.PointStruct(
        id=make_point_id(chunk_id),
        vector={VECTOR_NAME: embedding_record["embedding"]},
        payload=payload,
    )


def upload_embeddings(
    client: QdrantClient,
    path: Path,
    chunks: dict[str, dict[str, Any]],
    batch_size: int,
) -> int:
    """Stream validated embeddings into Qdrant using batched idempotent upserts.

    Args:
        client: Connected Qdrant client.
        path: Embedding JSONL file.
        chunks: Source chunks indexed by chunk_id.
        batch_size: Maximum number of points per upsert request.

    Returns:
        Total number of points submitted to Qdrant.

    Notes:
        Deterministic point IDs mean that rerunning this function updates
        existing points rather than creating duplicate points.
    """

    batch: list[models.PointStruct] = []
    uploaded_count = 0

    for record in read_jsonl(path):
        # Recheck the source relationship while reading the upload pass.
        # This also detects accidental input changes between the two passes.
        chunk_id = validate_embedding_record(record, chunks)
        point = build_point(record, chunks[chunk_id])
        batch.append(point)

        if len(batch) >= batch_size:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True,
            )

            uploaded_count += len(batch)
            LOGGER.info(
                "Upserted %d / %d points.",
                uploaded_count,
                len(chunks),
            )
            batch.clear()

    if batch:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True,
        )

        uploaded_count += len(batch)
        LOGGER.info(
            "Upserted %d / %d points.",
            uploaded_count,
            len(chunks),
        )

    return uploaded_count


# ---------------------------------------------------------------------------
# Command-Line Interface
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for the ingestion pipeline."""

    parser = argparse.ArgumentParser(
        description="Validate and ingest SciLitRAG embeddings into Qdrant."
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to the source chunks JSONL file.",
    )

    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to the embeddings JSONL file.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of points per Qdrant upsert batch.",
    )

    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=QDRANT_URL,
        help="Qdrant server URL.",
    )

    return parser.parse_args()


def main() -> int:
    """Run validation, ingestion, and final point-count verification."""

    configure_logging()
    args = parse_arguments()

    if args.batch_size < 1:
        LOGGER.error("Batch size must be greater than zero.")
        return 1

    client: QdrantClient | None = None

    try:
        LOGGER.info("=" * 65)
        LOGGER.info("SCILITRAG QDRANT INGESTION")
        LOGGER.info("=" * 65)

        # Load the smaller chunk dataset once for text-hash verification
        # and payload construction during the streaming embedding passes.
        chunks = load_chunks(args.chunks)

        # Validate the complete embedding dataset before making any writes.
        expected_count = validate_all_embeddings(args.embeddings, chunks)

        client = QdrantClient(url=args.qdrant_url, timeout=60)
        LOGGER.info("Connected to Qdrant at %s.", args.qdrant_url)

        verify_collection(client)

        LOGGER.info("Starting batched ingestion (batch size: %d).", args.batch_size)

        uploaded_count = upload_embeddings(
            client=client,
            path=args.embeddings,
            chunks=chunks,
            batch_size=args.batch_size,
        )

        if uploaded_count != expected_count:
            raise ValueError(
                f"Upload count mismatch: submitted {uploaded_count}, "
                f"expected {expected_count}."
            )

        # Request an exact count rather than relying on collection metadata.
        final_count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        if final_count != expected_count:
            raise ValueError(
                f"Qdrant point-count mismatch: expected {expected_count}, "
                f"found {final_count}."
            )

        LOGGER.info("=" * 65)
        LOGGER.info("QDRANT INGESTION COMPLETED SUCCESSFULLY")
        LOGGER.info("Collection: %s", COLLECTION_NAME)
        LOGGER.info("Points submitted: %d", uploaded_count)
        LOGGER.info("Verified Qdrant points: %d", final_count)
        LOGGER.info("Vector dimensions: %d", EXPECTED_DIMENSIONS)
        LOGGER.info("Embedding model: %s", EXPECTED_MODEL)
        LOGGER.info("=" * 65)

        return 0

    except Exception:
        LOGGER.exception("Qdrant ingestion failed.")
        return 1

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
