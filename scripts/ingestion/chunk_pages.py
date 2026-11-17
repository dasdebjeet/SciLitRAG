
"""
Chunk extracted research-paper pages into token-aware text segments.

The script reads page-level JSONL records pages.jsonl produced by extract_text.py,
splits their text into sentence-aware chunks, and writes the resulting
chunks to a separate JSONL file.

Design decisions:
    - Uses the BGE-M3 tokenizer for accurate token counting.
    - Processes each page independently to preserve precise page citations.
    - Targets a configurable chunk size with configurable token overlap.
    - Splits exceptionally long sentences when they exceed the chunk limit.
    - Generates deterministic chunk IDs for reproducibility.
    - Writes to a temporary file before replacing the final output.

Limitations:
    - Sentence boundaries are detected using a lightweight punctuation-based
      splitter. Scientific abbreviations and equations may occasionally
      produce imperfect sentence boundaries.
    - Page-level chunking does not combine text across consecutive pages.
"""

from __future__ import annotations
from scilitrag.core.paths import PROCESSED_DATA_DIR

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------



DEFAULT_INPUT = PROCESSED_DATA_DIR / "pages.jsonl"
DEFAULT_OUTPUT = PROCESSED_DATA_DIR / "chunks.jsonl"

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 80

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure consistent console logging for the chunking pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def read_jsonl(file_path: Path) -> Iterator[dict[str, Any]]:
    """
    Read JSON objects from a JSONL file.

    Args:
        file_path: Path to the input JSONL file.

    Yields:
        One parsed dictionary per non-empty line.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If a line contains invalid JSON or is not an object.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at line {line_number}."
                )

            yield record


# ---------------------------------------------------------------------------
# Sentence and token processing
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> list[str]:
    """
    Split page text into approximate sentences.

    The splitter preserves punctuation and uses sentence-ending punctuation
    followed by whitespace as its primary boundary.

    Args:
        text: Extracted page text.

    Returns:
        A list of non-empty sentence-like text segments.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def encode_text(tokenizer: Any, text: str) -> list[int]:
    """
    Tokenize text without adding model-specific special tokens.

    Special tokens are normally added when preparing final model inputs.
    They are excluded here so chunk-size measurements represent content
    tokens rather than tokenizer-added overhead.
    """
    return tokenizer.encode(
        text,
        add_special_tokens=False,
    )


def decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    """Convert token IDs back into readable text."""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()


def chunk_page_text(
    text: str,
    tokenizer: Any,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int]]:
    """
    Generate sentence-aware, token-limited chunks from one PDF page.

    Complete sentences are kept together whenever possible. If a sentence
    cannot fit within the remaining chunk capacity, it is split at the
    tokenizer's token boundary.

    After emitting a chunk, its final `overlap` tokens are carried into
    the next chunk to retain some local context.

    Args:
        text: Extracted text from a single PDF page.
        tokenizer: Loaded BGE-M3 tokenizer.
        chunk_size: Maximum content tokens per chunk.
        overlap: Number of trailing tokens shared between chunks.

    Returns:
        A list of (chunk_text, token_count) tuples.
    """
    if not text or not text.strip():
        return []

    sentences = split_into_sentences(text)

    chunks: list[tuple[str, int]] = []
    current_tokens: list[int] = []

    def emit_chunk() -> None:
        """Save the current chunk and retain its trailing overlap."""
        nonlocal current_tokens

        if not current_tokens:
            return

        chunk_text = decode_tokens(tokenizer, current_tokens)

        if chunk_text:
            chunks.append((chunk_text, len(current_tokens)))

        # Retain only the trailing context needed by the next chunk.
        current_tokens = (
            current_tokens[-overlap:] if overlap > 0 else []
        )

    for sentence in sentences:
        sentence_tokens = encode_text(tokenizer, sentence)

        if not sentence_tokens:
            continue

        # A sentence may be longer than the entire chunk budget.
        # Process it in token-sized pieces rather than dropping its content.
        remaining_tokens = sentence_tokens

        while remaining_tokens:
            available_capacity = chunk_size - len(current_tokens)

            if available_capacity <= 0:
                emit_chunk()
                available_capacity = chunk_size - len(current_tokens)

            tokens_to_add = remaining_tokens[:available_capacity]
            current_tokens.extend(tokens_to_add)
            remaining_tokens = remaining_tokens[available_capacity:]

            # If a sentence continues beyond the available capacity,
            # emit the current chunk and continue with the remaining tokens.
            if remaining_tokens:
                emit_chunk()

        # If the next sentence cannot fit, emit the current chunk first.
        # The next iteration will continue with the carried overlap.

    # Emit the final chunk. Avoid creating a duplicate of the previous one.
    if current_tokens:
        final_text = decode_tokens(tokenizer, current_tokens)

        if final_text and (
            not chunks or final_text != chunks[-1][0]
        ):
            chunks.append((final_text, len(current_tokens)))

    return chunks


# ---------------------------------------------------------------------------
# Chunk metadata
# ---------------------------------------------------------------------------

def generate_chunk_id(
    paper_id: str,
    page_number: int,
    chunk_index: int,
) -> str:
    """
    Generate a deterministic identifier for a page-level chunk.

    The identifier remains stable as long as the paper ID, page number,
    and chunk index remain unchanged.
    """
    source = f"{paper_id}:page:{page_number}:chunk:{chunk_index}"

    return hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()[:16]


def create_chunk_records(
    page: dict[str, Any],
    tokenizer: Any,
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """
    Convert one extracted page record into chunk records.

    Each output record preserves the paper and page provenance required
    for retrieval results and source citations.

    Raises:
        ValueError: If required page metadata is missing.
    """
    paper_id = page.get("arxiv_id") or page.get("paper_id")
    title = page.get("title")
    page_number = page.get("page_number")
    text = page.get("text", "")

    if not paper_id or not title or page_number is None:
        raise ValueError(
            "Page record is missing required metadata: "
            "arxiv_id/paper_id, title, or page_number."
        )

    page_chunks = chunk_page_text(
        text=text,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    records = []

    for chunk_index, (chunk_text, token_count) in enumerate(
        page_chunks,
        start=0,
    ):
        records.append({
            "chunk_id": generate_chunk_id(
                str(paper_id),
                int(page_number),
                chunk_index,
            ),
            "arxiv_id": paper_id,
            "title": title,
            "page_number": page_number,
            "total_pages": page.get("total_pages"),
            "chunk_index": chunk_index,
            "text": chunk_text,
            "token_count": token_count,
            "char_count": len(chunk_text),
            "pdf_path": page.get("pdf_path"),
        })

    return records


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line configuration for the chunking pipeline."""
    parser = argparse.ArgumentParser(
        description="Create token-aware chunks from extracted PDF pages."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input page-level JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output chunk-level JSONL file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face tokenizer model identifier.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum content tokens per chunk.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Trailing tokens carried into the next chunk.",
    )

    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be greater than zero.")

    if args.overlap < 0 or args.overlap >= args.chunk_size:
        parser.error(
            "--overlap must be non-negative and smaller than --chunk-size."
        )

    return args


def main() -> None:
    """Run the complete page-to-chunk processing pipeline."""
    configure_logging()
    args = parse_arguments()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")

    logger.info("=" * 65)
    logger.info("SCILITRAG CHUNKING PIPELINE")
    logger.info("=" * 65)
    logger.info("Input: %s", input_path)
    logger.info("Output: %s", output_path)
    logger.info("Tokenizer: %s", args.model)
    logger.info("Chunk size: %d tokens", args.chunk_size)
    logger.info("Overlap: %d tokens", args.overlap)

    logger.info("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for this pipeline."
        )

    logger.info("Tokenizer loaded successfully.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write to a temporary file so an interrupted run does not leave
    # a partially written file at the final output location.
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    total_pages = 0
    total_chunks = 0
    failed_pages = 0
    total_tokens = 0

    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for page in read_jsonl(input_path):
                total_pages += 1

                try:
                    page_chunks = create_chunk_records(
                        page=page,
                        tokenizer=tokenizer,
                        chunk_size=args.chunk_size,
                        overlap=args.overlap,
                    )

                    for chunk in page_chunks:
                        output_file.write(
                            json.dumps(
                                chunk,
                                ensure_ascii=False,
                            ) + "\n"
                        )

                    total_chunks += len(page_chunks)
                    total_tokens += sum(
                        chunk["token_count"]
                        for chunk in page_chunks
                    )

                except (ValueError, TypeError, KeyError) as exc:
                    failed_pages += 1
                    logger.exception(
                        "Failed to process paper %s, page %s: %s",
                        page.get("arxiv_id", "unknown"),
                        page.get("page_number", "unknown"),
                        exc,
                    )

            output_file.flush()

        # Replace the previous output only after successful completion.
        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    average_tokens = (
        total_tokens / total_chunks if total_chunks else 0
    )

    logger.info("=" * 65)
    logger.info("CHUNKING SUMMARY")
    logger.info("Total pages processed: %d", total_pages)
    logger.info("Total chunks generated: %d", total_chunks)
    logger.info("Failed pages: %d", failed_pages)
    logger.info("Total chunk tokens: %d", total_tokens)
    logger.info("Average chunk tokens: %.2f", average_tokens)
    logger.info("Output: %s", output_path)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
