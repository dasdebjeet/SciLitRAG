
"""Build and query an in-memory BM25 index for SciLitRAG.

Responsibilities:
    - Load research-paper chunks from the processed JSONL dataset.
    - Tokenize scientific text while preserving useful technical terminology.
    - Build a BM25Okapi lexical retrieval index.
    - Rank chunks against natural-language research queries.
    - Return original chunk metadata for traceability and citations.

Design:
    BM25 is maintained separately from Qdrant's dense vector index.
    The two retrieval result lists will be combined later using Reciprocal
    Rank Fusion (RRF).

Limitations:
    - The index is rebuilt in memory whenever this script starts.
    - Tokenization is intentionally simple and language-agnostic.
    - BM25 scores are ranking scores, not calibrated relevance probabilities.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from scilitrag.core.paths import DEFAULT_CHUNKS_PATH


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


DEFAULT_TOP_K = 5

LOGGER = logging.getLogger(__name__)

# Preserve technical terms containing hyphens, dots, underscores, and digits.
# Examples: ResNet-50, YOLOv8, BERT, 3.14, model_v2.
#
# Also retain comparison and mathematical operators as separate tokens.
TOKEN_PATTERN = re.compile(
    r"(?u)[\w]+(?:[.-][\w]+)*|[<>]=?|[=±×÷]+"
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure consistent timestamped logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Convert scientific text into lowercase BM25 tokens.

    Args:
        text: Source text or search query.

    Returns:
        A list of normalized tokens.

    Notes:
        This tokenizer deliberately avoids aggressive stemming and stop-word
        removal. In scientific literature, common words and small tokens can
        be meaningful in equations, abbreviations, and technical expressions.
    """

    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
    ]


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Load and validate research-paper chunks from a JSONL file.

    Args:
        path: Path to the processed chunks JSONL file.

    Returns:
        A list of chunk records in their original dataset order.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If a record is malformed or contains duplicate IDs.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Chunk dataset not found: {path}")

    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
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

            chunk_id = record.get("chunk_id")
            text = record.get("text")

            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(
                    f"Missing or invalid chunk_id at line {line_number}."
                )

            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk ID: {chunk_id}")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Missing or empty text for chunk {chunk_id}."
                )

            required_fields = ("arxiv_id", "title", "page_number")

            missing_fields = [
                field for field in required_fields if field not in record
            ]

            if missing_fields:
                raise ValueError(
                    f"Chunk {chunk_id} is missing fields: {missing_fields}"
                )

            seen_ids.add(chunk_id)
            chunks.append(record)

    if not chunks:
        raise ValueError(f"No valid chunks found in {path}.")

    LOGGER.info("Loaded %d chunks from %s.", len(chunks), path)

    return chunks


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

class BM25Retriever:
    """In-memory lexical retrieval engine backed by BM25Okapi."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        """Initialize the BM25 index from validated chunk records.

        Args:
            chunks: Source chunks containing text and paper metadata.

        Raises:
            ValueError: If the dataset is empty or contains no searchable text.
        """

        if not chunks:
            raise ValueError("Cannot build a BM25 index from an empty dataset.")

        self.chunks = chunks

        LOGGER.info("Tokenizing %d chunks for BM25 indexing.", len(chunks))

        self.tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        empty_documents = sum(
            not tokens for tokens in self.tokenized_corpus
        )

        if empty_documents == len(self.tokenized_corpus):
            raise ValueError("All chunks produced empty token sequences.")

        if empty_documents:
            LOGGER.warning(
                "%d chunks produced no searchable tokens.",
                empty_documents,
            )

        LOGGER.info("Building BM25Okapi index.")

        self.index = BM25Okapi(self.tokenized_corpus)

        LOGGER.info("BM25 index built successfully.")

    
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        arxiv_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the highest-scoring chunks for a lexical query.

        Args:
            query: Natural-language or keyword-based search query.
            top_k: Maximum number of results to return.
            arxiv_id: Optional arXiv ID restricting retrieval to one paper.

        Returns:
            Ranked result dictionaries containing BM25 scores and original
            chunk metadata.

        Raises:
            ValueError: If the query is empty or top_k is invalid.
        """

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = tokenize(query)

        if not query_tokens:
            raise ValueError("Search query produced no searchable tokens.")

        scores = self.index.get_scores(query_tokens)

        # Apply the paper filter before selecting top results. Filtering only
        # after top-k selection could exclude relevant chunks from that paper.
        ranked_indices = sorted(
            (
                index
                for index, score in enumerate(scores)
                if score > 0
                and (
                    arxiv_id is None
                    or self.chunks[index].get("arxiv_id") == arxiv_id
                )
            ),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[dict[str, Any]] = []

        for rank, index in enumerate(ranked_indices[:top_k], start=1):
            chunk = self.chunks[index]

            results.append(
                {
                    "rank": rank,
                    "score": float(scores[index]),
                    "chunk_id": chunk["chunk_id"],
                    "arxiv_id": chunk["arxiv_id"],
                    "title": chunk["title"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                    "metadata": chunk,
                }
            )

        return results



# ---------------------------------------------------------------------------
# Command-Line Interface
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse the BM25 search command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Search SciLitRAG research chunks using BM25."
    )

    parser.add_argument(
        "--query",
        type=str,
        default=(
            "How do machine learning methods quantify "
            "pore-space evolution over time?"
        ),
        help="Research question or keyword-based search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Maximum number of results to display.",
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to the source chunks JSONL file.",
    )

    return parser.parse_args()


def main() -> int:
    """Load the corpus, build the BM25 index, and execute a search."""

    configure_logging()
    args = parse_arguments()

    if not args.query.strip():
        LOGGER.error("Search query cannot be empty.")
        return 1

    if args.top_k < 1:
        LOGGER.error("top-k must be greater than zero.")
        return 1

    try:
        chunks = load_chunks(args.chunks)
        retriever = BM25Retriever(chunks)

        LOGGER.info("Executing BM25 search.")

        results = retriever.search(
            query=args.query,
            top_k=args.top_k,
        )

        print("\n" + "=" * 85)
        print("SCILITRAG — BM25 LEXICAL SEARCH")
        print("=" * 85)
        print(f"Query: {args.query}")
        print(f"Results returned: {len(results)}")

        if not results:
            print("\nNo documents with lexical overlap were found.")
            return 0

        for result in results:
            print("\n" + "-" * 85)
            print(
                f"RANK: {result['rank']} | "
                f"BM25 SCORE: {result['score']:.6f}"
            )
            print(f"Title: {result['title']}")
            print(f"arXiv ID: {result['arxiv_id']}")
            print(f"Page: {result['page_number']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print("\nRetrieved text:")
            print(result["text"][:1200])

            if len(result["text"]) > 1200:
                print("... [excerpt truncated]")

        print("\n" + "=" * 85)

        return 0

    except Exception:
        LOGGER.exception("BM25 retrieval failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
