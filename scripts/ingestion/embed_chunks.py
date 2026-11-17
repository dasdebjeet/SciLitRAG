
"""
Generate dense vector embeddings for SciLitRAG document chunks.

Responsibilities:
    - Read chunk records from a JSONL file.
    - Generate normalized embeddings using Sentence Transformers.
    - Use GPU acceleration when available.
    - Persist embeddings incrementally for safe recovery.
    - Validate existing embeddings using chunk IDs and SHA256 text hashes.
    - Discard stale, invalid, or obsolete checkpoint records.
    - Recover from CUDA out-of-memory errors by reducing the encoding batch size.

Output schema:
    {
        "chunk_id": "...",
        "text_hash": "...",
        "model": "BAAI/bge-m3",
        "dimensions": 1024,
        "normalized": true,
        "embedding": [...]
    }

Important:
    The text hash is calculated from the exact source text supplied to the
    embedding model. Any text modification therefore invalidates its old
    embedding, even when the chunk ID remains unchanged.
"""

from __future__ import annotations
from scilitrag.core.paths import PROCESSED_DATA_DIR

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

import argparse
import hashlib
import json
import logging
import os
import tempfile

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

import numpy as np
import torch

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------



DEFAULT_INPUT = PROCESSED_DATA_DIR / "chunks.jsonl"
DEFAULT_OUTPUT = PROCESSED_DATA_DIR / "embeddings.jsonl"

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DEVICE = "cuda"
DEFAULT_BATCH_SIZE = 16
DEFAULT_DIMENSIONS = 1024

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure consistent console logging for the embedding pipeline."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Input validation and hashing
# ---------------------------------------------------------------------------

def calculate_text_hash(text: str) -> str:
    """
    Calculate a SHA256 fingerprint of the exact source text.

    The hash allows the pipeline to detect changes to chunk content even
    when the chunk ID itself has not changed.

    Args:
        text: Source text supplied to the embedding model.

    Returns:
        A hexadecimal SHA256 digest.
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_input_hashes(
    input_path: Path,
    limit: int | None = None,
) -> dict[str, str]:
    """
    Read chunk IDs and source-text hashes from the input JSONL file.

    Only lightweight metadata is retained in memory. The actual chunk text
    is read again later in batches to avoid loading the complete dataset.

    Args:
        input_path: JSONL file containing extracted document chunks.
        limit: Optional maximum number of chunks for smoke testing.

    Returns:
        Mapping of chunk IDs to their source-text SHA256 hashes.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If a record is malformed or contains duplicate IDs.
    """

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    input_hashes: dict[str, str] = {}

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):

            if limit is not None and len(input_hashes) >= limit:
                break

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in input file at line {line_number}"
                ) from exc

            chunk_id = record.get("chunk_id")
            text = record.get("text")

            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(
                    f"Missing or invalid chunk_id at line {line_number}"
                )

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Missing or empty chunk text at line {line_number}"
                )

            if chunk_id in input_hashes:
                raise ValueError(
                    f"Duplicate chunk_id in input file: {chunk_id}"
                )

            input_hashes[chunk_id] = calculate_text_hash(text)

    if not input_hashes:
        raise ValueError("No valid chunks were found in the input file.")

    LOGGER.info("Loaded %s input chunk fingerprints.", len(input_hashes))

    return input_hashes


# ---------------------------------------------------------------------------
# Checkpoint validation and recovery
# ---------------------------------------------------------------------------

def is_valid_embedding_record(
    record: dict[str, Any],
    expected_hash: str,
    model_name: str,
    dimensions: int,
) -> bool:
    """
    Check whether an existing embedding can safely be reused.

    A checkpoint is reusable only when its source hash, model, dimensions,
    normalization status, and vector contents are valid.

    Args:
        record: Existing embedding JSON record.
        expected_hash: Current source-text hash for this chunk.
        model_name: Expected embedding model identifier.
        dimensions: Expected embedding vector dimension.

    Returns:
        True when the record is safe to reuse; otherwise False.
    """

    if record.get("text_hash") != expected_hash:
        return False

    if record.get("model") != model_name:
        return False

    if record.get("dimensions") != dimensions:
        return False

    if record.get("normalized") is not True:
        return False

    embedding = record.get("embedding")

    if not isinstance(embedding, list) or len(embedding) != dimensions:
        return False

    try:
        vector = np.asarray(embedding, dtype=np.float32)
    except (TypeError, ValueError):
        return False

    return bool(np.isfinite(vector).all())


def reconcile_checkpoint(
    output_path: Path,
    input_hashes: dict[str, str],
    model_name: str,
    dimensions: int,
) -> set[str]:
    """
    Retain valid checkpoint records and discard stale records.

    This operation is performed before embedding starts. It ensures that:
        - Unchanged chunks retain their existing embeddings.
        - Modified chunks are re-embedded.
        - Removed chunks are discarded.
        - Invalid or duplicate checkpoint records are discarded.

    The checkpoint is rewritten through a temporary file and atomically
    replaced, preventing a partially written reconciliation from replacing
    the original checkpoint.

    Args:
        output_path: Existing embedding JSONL checkpoint.
        input_hashes: Current chunk ID to text-hash mapping.
        model_name: Expected embedding model identifier.
        dimensions: Expected embedding dimension.

    Returns:
        Set of chunk IDs whose embeddings remain valid.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not output_path.exists():
        LOGGER.info("No existing checkpoint found. Starting fresh.")
        return set()

    completed_ids: set[str] = set()
    retained_count = 0
    discarded_count = 0

    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:

            temp_path = Path(temp_file.name)

            with output_path.open("r", encoding="utf-8") as checkpoint:

                for line_number, line in enumerate(checkpoint, start=1):

                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        discarded_count += 1
                        LOGGER.warning(
                            "Discarding malformed checkpoint line %s.",
                            line_number,
                        )
                        continue

                    chunk_id = record.get("chunk_id")

                    if not isinstance(chunk_id, str):
                        discarded_count += 1
                        continue

                    expected_hash = input_hashes.get(chunk_id)

                    if expected_hash is None:
                        discarded_count += 1
                        continue

                    if chunk_id in completed_ids:
                        discarded_count += 1
                        LOGGER.warning(
                            "Discarding duplicate checkpoint ID: %s",
                            chunk_id,
                        )
                        continue

                    if not is_valid_embedding_record(
                        record=record,
                        expected_hash=expected_hash,
                        model_name=model_name,
                        dimensions=dimensions,
                    ):
                        discarded_count += 1
                        continue

                    # Preserve the original valid JSON record without
                    # loading all embedding vectors into memory.
                    temp_file.write(json.dumps(record) + "\n")

                    completed_ids.add(chunk_id)
                    retained_count += 1

        os.replace(temp_path, output_path)
        temp_path = None

    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    LOGGER.info("Checkpoint reconciliation complete.")
    LOGGER.info("Valid embeddings retained: %s", retained_count)
    LOGGER.info("Stale/invalid records discarded: %s", discarded_count)

    return completed_ids


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def load_embedding_model(
    model_name: str,
    device: str,
    expected_dimensions: int,
) -> SentenceTransformer:
    """
    Load the embedding model and validate its output dimension.

    Args:
        model_name: Hugging Face model identifier.
        device: Requested execution device (CUDA or CPU).
        expected_dimensions: Expected embedding vector dimension.

    Returns:
        Initialized SentenceTransformer model.

    Raises:
        RuntimeError: If the model produces an unexpected dimension.
    """

    LOGGER.info("Loading embedding model: %s", model_name)
    LOGGER.info("Execution device: %s", device)

    model = SentenceTransformer(model_name, device=device)

    actual_dimensions = model.get_embedding_dimension()

    if actual_dimensions != expected_dimensions:
        raise RuntimeError(
            f"Unexpected embedding dimension: {actual_dimensions}. "
            f"Expected: {expected_dimensions}."
        )

    LOGGER.info("Embedding dimensions: %s", actual_dimensions)

    return model


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def encode_batch(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, int]:
    """
    Generate normalized embeddings with automatic OOM recovery.

    If the GPU runs out of memory, the internal model-encoding batch size
    is reduced and the operation is retried. The outer input batch remains
    unchanged, so all records are still processed together.

    Args:
        model: Initialized SentenceTransformer model.
        texts: Source texts to embed.
        batch_size: Initial internal encoding batch size.
        device: Execution device.

    Returns:
        Tuple containing the embedding matrix and successful batch size.

    Raises:
        RuntimeError: If encoding fails for a reason other than OOM, or
            if even a batch size of one cannot fit in memory.
    """

    current_batch_size = batch_size

    while True:
        try:
            embeddings = model.encode(
                texts,
                batch_size=current_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            return np.asarray(embeddings), current_batch_size

        except RuntimeError as exc:

            is_oom = (
                "out of memory" in str(exc).lower()
                or (
                    hasattr(torch.cuda, "OutOfMemoryError")
                    and isinstance(exc, torch.cuda.OutOfMemoryError)
                )
            )

            if not is_oom:
                raise

            if current_batch_size <= 1:
                raise RuntimeError(
                    "Out of memory even with an encoding batch size of 1. "
                    "Try CPU execution or a smaller embedding model."
                ) from exc

            current_batch_size = max(1, current_batch_size // 2)

            LOGGER.warning(
                "Memory allocation failed. Retrying with batch size %s.",
                current_batch_size,
            )

            if device.startswith("cuda"):
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------

def append_embeddings(
    output_path: Path,
    records: list[dict[str, Any]],
) -> None:
    """
    Append a completed embedding batch to the checkpoint.

    Each batch is flushed before processing continues. Therefore, if the
    process is interrupted, previously completed batches remain available
    for recovery.

    Args:
        output_path: Destination embedding JSONL file.
        records: Completed embedding records.
    """

    with output_path.open("a", encoding="utf-8") as output_file:

        for record in records:
            output_file.write(
                json.dumps(record, allow_nan=False) + "\n"
            )

        output_file.flush()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def embed_chunks(args: argparse.Namespace) -> None:
    """
    Run the complete chunk embedding pipeline.

    The pipeline first reconciles the existing checkpoint, then streams
    missing chunks from the input file and persists their embeddings in
    batches.

    Args:
        args: Parsed command-line arguments.
    """

    input_hashes = load_input_hashes(
        input_path=args.input,
        limit=args.limit,
    )

    completed_ids = reconcile_checkpoint(
        output_path=args.output,
        input_hashes=input_hashes,
        model_name=args.model,
        dimensions=args.dimensions,
    )

    if len(completed_ids) == len(input_hashes):
        LOGGER.info("All input chunks already have valid embeddings.")
        return

    model = load_embedding_model(
        model_name=args.model,
        device=args.device,
        expected_dimensions=args.dimensions,
    )

    total_chunks = len(input_hashes)
    pending_count = total_chunks - len(completed_ids)

    LOGGER.info("Total input chunks: %s", total_chunks)
    LOGGER.info("Already embedded: %s", len(completed_ids))
    LOGGER.info("Remaining chunks: %s", pending_count)

    current_batch: list[dict[str, Any]] = []
    processed_count = 0
    current_encode_batch_size = args.batch_size

    def process_batch(batch: list[dict[str, Any]]) -> None:
        """Embed one outer batch and persist its completed records."""

        nonlocal current_encode_batch_size
        nonlocal processed_count

        texts = [record["text"] for record in batch]

        embeddings, current_encode_batch_size = encode_batch(
            model=model,
            texts=texts,
            batch_size=current_encode_batch_size,
            device=args.device,
        )

        if embeddings.shape != (len(batch), args.dimensions):
            raise RuntimeError(
                f"Unexpected embedding matrix shape: {embeddings.shape}"
            )

        if not np.isfinite(embeddings).all():
            raise RuntimeError("Embedding matrix contains NaN or infinity.")

        output_records = []

        for source_record, embedding in zip(batch, embeddings):

            chunk_id = source_record["chunk_id"]
            text = source_record["text"]

            output_records.append(
                {
                    "chunk_id": chunk_id,
                    "text_hash": calculate_text_hash(text),
                    "model": args.model,
                    "dimensions": args.dimensions,
                    "normalized": True,
                    "embedding": embedding.tolist(),
                }
            )

        append_embeddings(args.output, output_records)

        processed_count += len(batch)

        LOGGER.info(
            "Progress: %s/%s pending chunks embedded.",
            processed_count,
            pending_count,
        )

    with args.input.open("r", encoding="utf-8") as input_file:

        for line_number, line in enumerate(input_file, start=1):

            if not line.strip():
                continue

            record = json.loads(line)
            chunk_id = record["chunk_id"]

            # Respect the optional smoke-test limit.
            if chunk_id not in input_hashes:
                if args.limit is not None and len(input_hashes) <= args.limit:
                    continue

            if chunk_id in completed_ids:
                continue

            current_batch.append(record)

            if len(current_batch) >= args.batch_size:
                process_batch(current_batch)
                current_batch = []

        if current_batch:
            process_batch(current_batch)

    final_count = len(
        reconcile_checkpoint(
            output_path=args.output,
            input_hashes=input_hashes,
            model_name=args.model,
            dimensions=args.dimensions,
        )
    )

    if final_count != total_chunks:
        raise RuntimeError(
            f"Embedding count mismatch: expected {total_chunks}, "
            f"found {final_count}."
        )

    LOGGER.info("=" * 65)
    LOGGER.info("EMBEDDING PIPELINE COMPLETED")
    LOGGER.info("Total valid embeddings: %s", final_count)
    LOGGER.info("Output: %s", args.output)
    LOGGER.info("=" * 65)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for the embedding pipeline."""

    parser = argparse.ArgumentParser(
        description="Generate BGE-M3 embeddings for SciLitRAG chunks."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input chunk JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output embedding JSONL file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Sentence Transformers model identifier.",
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=DEFAULT_DEVICE,
        help="Execution device.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Initial encoding and outer batch size.",
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help="Expected embedding dimension.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of input chunks for smoke testing.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    if args.dimensions < 1:
        parser.error("--dimensions must be at least 1.")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")

    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output paths must be different.")

    return args


def main() -> None:
    """Initialize logging and execute the embedding pipeline."""

    configure_logging()
    args = parse_arguments()

    try:
        embed_chunks(args)

    except KeyboardInterrupt:
        LOGGER.warning(
            "Embedding interrupted. Completed batches remain checkpointed."
        )
        raise SystemExit(130)

    except Exception:
        LOGGER.exception("Embedding pipeline failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
