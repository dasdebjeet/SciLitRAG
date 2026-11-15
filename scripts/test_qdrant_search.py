
"""Run a semantic similarity search against the SciLitRAG Qdrant collection.

The query is encoded using the same BGE-M3 model used during ingestion.
Qdrant then compares the query vector against the stored dense vectors
and returns the most similar chunks with their source metadata.
"""

from __future__ import annotations

import argparse
import logging

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "scilit_rag"
VECTOR_NAME = "dense"

MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda"
DEFAULT_TOP_K = 5

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging and CLI
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure timestamped logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse the search query and retrieval count."""

    parser = argparse.ArgumentParser(
        description="Test semantic retrieval against SciLitRAG."
    )

    parser.add_argument(
        "--query",
        type=str,
        default=(
            "How can machine learning models quantify changes "
            "in pore-space structure over time?"
        ),
        help="Natural-language research question.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of matching chunks to retrieve.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Semantic Search
# ---------------------------------------------------------------------------

def main() -> int:
    """Encode the query, retrieve matching chunks, and display evidence."""

    configure_logging()
    args = parse_arguments()

    if not args.query.strip():
        LOGGER.error("Search query cannot be empty.")
        return 1

    if args.top_k < 1:
        LOGGER.error("top-k must be greater than zero.")
        return 1

    client: QdrantClient | None = None

    try:
        LOGGER.info("Loading embedding model: %s", MODEL_NAME)

        model = SentenceTransformer(MODEL_NAME, device=DEVICE, local_files_only=True)

        # Query encoding must use the same model and normalization
        # configuration as the document embeddings stored in Qdrant.
        query_vector = model.encode(
            args.query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        LOGGER.info("Connecting to Qdrant: %s", QDRANT_URL)
        client = QdrantClient(url=QDRANT_URL, timeout=30)

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            using=VECTOR_NAME,
            limit=args.top_k,
            with_payload=True,
        )

        print("\n" + "=" * 80)
        print("SCILITRAG SEMANTIC SEARCH")
        print("=" * 80)
        print(f"Query: {args.query}")
        print(f"Results requested: {args.top_k}")
        print(f"Results returned: {len(response.points)}")

        if not response.points:
            print("\nNo matching chunks found.")
            return 0

        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}

            print("\n" + "-" * 80)
            print(f"RANK: {rank} | SCORE: {point.score:.6f}")
            print(f"Title: {payload.get('title', 'Unknown')}")
            print(f"arXiv ID: {payload.get('arxiv_id', 'Unknown')}")
            print(f"Page: {payload.get('page_number', 'Unknown')}")
            print(f"Chunk ID: {payload.get('chunk_id', 'Unknown')}")
            print("\nRetrieved text:")

            text = payload.get("text", "")
            print(text[:1200])

            if len(text) > 1200:
                print("... [excerpt truncated]")

        print("\n" + "=" * 80)
        return 0

    except Exception:
        LOGGER.exception("Semantic search failed.")
        return 1

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
