
"""
Dense semantic retrieval for SciLitRAG.

This module embeds natural-language queries with BAAI/bge-m3 and searches
the existing Qdrant collection using cosine similarity.

Responsibilities:
    - Load and validate the embedding model.
    - Verify that the Qdrant collection has the expected named vector.
    - Encode queries using the same normalization strategy as corpus embeddings.
    - Retrieve passages with optional arXiv ID filtering.
    - Return a standardized result structure for downstream hybrid retrieval.

Assumptions:
    - Qdrant is running and reachable at the configured URL.
    - The collection has already been populated by ingest_qdrant.py.
    - The collection contains a named dense vector called "dense".
    - Stored vectors were generated using BAAI/bge-m3 with 1024 dimensions.

Limitations:
    - Results are ranked by cosine similarity, not answer relevance.
    - Similarity scores are not probabilities.
    - This module does not perform lexical retrieval or reranking.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import numpy as np
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "scilit_rag"
DEFAULT_MODEL = "BAAI/bge-m3"

VECTOR_NAME = "dense"
EXPECTED_DIMENSIONS = 1024

DEFAULT_TOP_K = 5
DEFAULT_DEVICE = "cuda"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dense Retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Retrieve semantically relevant paper passages from Qdrant.

    The model and Qdrant client are initialized once per instance, allowing
    multiple queries to reuse the same resources without repeatedly loading
    the embedding model or reconnecting to the database.

    Args:
        qdrant_url: URL of the running Qdrant service.
        collection_name: Existing Qdrant collection to search.
        model_name: SentenceTransformer model used for query embeddings.
        device: Inference device, normally "cuda" or "cpu".

    Raises:
        RuntimeError: If the model or Qdrant collection cannot be initialized.
        ValueError: If the collection's dense vector configuration is invalid.
    """

    def __init__(
        self,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self.collection_name = collection_name
        self.model_name = model_name
        self.device = device

        self.client: QdrantClient | None = None
        self.model: SentenceTransformer | None = None

        try:
            # Load the model once so subsequent searches reuse it.
            logger.info("Loading embedding model: %s", model_name)

            self.model = SentenceTransformer(
                model_name,
                device=device,
            )

            model_dimensions = self.model.get_embedding_dimension()

            if model_dimensions != EXPECTED_DIMENSIONS:
                raise ValueError(
                    f"Unexpected model dimensions: {model_dimensions}. "
                    f"Expected {EXPECTED_DIMENSIONS}."
                )

            logger.info(
                "Embedding model loaded | dimensions=%d | device=%s",
                model_dimensions,
                device,
            )

            # Connect to the existing Qdrant service.
            self.client = QdrantClient(url=qdrant_url)

            # Verify the collection rather than silently creating or replacing it.
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )

            vector_config = collection_info.config.params.vectors

            if not isinstance(vector_config, dict):
                raise ValueError(
                    "The collection does not use named vectors. "
                    f"Expected a named vector called '{VECTOR_NAME}'."
                )

            dense_config = vector_config.get(VECTOR_NAME)

            if dense_config is None:
                raise ValueError(
                    f"Named vector '{VECTOR_NAME}' was not found in "
                    f"collection '{self.collection_name}'."
                )

            if dense_config.size != EXPECTED_DIMENSIONS:
                raise ValueError(
                    f"Qdrant vector size is {dense_config.size}; "
                    f"expected {EXPECTED_DIMENSIONS}."
                )

            if dense_config.distance != models.Distance.COSINE:
                raise ValueError(
                    "The Qdrant dense vector must use cosine distance."
                )

            logger.info(
                "Qdrant connection verified | collection=%s | vector=%s | "
                "dimensions=%d | distance=cosine",
                self.collection_name,
                VECTOR_NAME,
                dense_config.size,
            )

        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"Failed to initialize DenseRetriever: {exc}"
            ) from exc

    def encode_query(self, query: str) -> list[float]:
        """
        Convert a query into a normalized dense embedding.

        Query normalization must match the normalization used when generating
        the stored corpus embeddings. This is important for cosine similarity.

        Args:
            query: Natural-language search query.

        Returns:
            A list containing 1024 normalized floating-point values.

        Raises:
            ValueError: If the query is empty or the vector is invalid.
        """

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        assert self.model is not None

        embedding = self.model.encode(
            query.strip(),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.shape != (EXPECTED_DIMENSIONS,):
            raise ValueError(
                f"Invalid query embedding shape: {embedding.shape}"
            )

        if not np.isfinite(embedding).all():
            raise ValueError("Query embedding contains non-finite values.")

        norm = float(np.linalg.norm(embedding))

        if not np.isclose(norm, 1.0, atol=1e-3):
            raise ValueError(
                f"Query embedding is not normalized. L2 norm={norm:.6f}"
            )

        return embedding.tolist()

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        arxiv_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most semantically similar passages.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of passages to return.
            arxiv_id: Optional arXiv ID used to restrict the search.

        Returns:
            Ranked passage records containing scores, identifiers, text,
            and original Qdrant payload metadata.

        Raises:
            ValueError: If query or top_k is invalid.
            RuntimeError: If the Qdrant search fails.
        """

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        assert self.client is not None

        query_vector = self.encode_query(query)

        # Apply metadata filtering inside Qdrant, before candidate ranking.
        query_filter = None

        if arxiv_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="arxiv_id",
                        match=models.MatchValue(value=arxiv_id),
                    )
                ]
            )

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=VECTOR_NAME,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Qdrant dense search failed: {exc}"
            ) from exc

        results: list[dict[str, Any]] = []

        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}

            results.append(
                {
                    "rank": rank,
                    "score": float(point.score),
                    "chunk_id": payload.get("chunk_id"),
                    "arxiv_id": payload.get("arxiv_id"),
                    "title": payload.get("title"),
                    "page_number": payload.get("page_number"),
                    "text": payload.get("text", ""),
                    "metadata": payload,
                }
            )

        logger.info(
            "Dense retrieval completed | query=%r | results=%d",
            query,
            len(results),
        )

        return results

    def close(self) -> None:
        """Release the Qdrant client connection if it was initialized."""

        if self.client is not None:
            self.client.close()
            self.client = None


# ---------------------------------------------------------------------------
# Command-Line Interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone dense retrieval."""

    parser = argparse.ArgumentParser(
        description="Search SciLitRAG using dense semantic retrieval."
    )

    parser.add_argument(
        "--query",
        type=str,
        default="How do machine learning methods quantify pore-space evolution over time?",
        help="Natural-language search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of passages to retrieve.",
    )

    parser.add_argument(
        "--arxiv-id",
        type=str,
        default=None,
        help="Optional arXiv ID filter.",
    )

    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=DEFAULT_QDRANT_URL,
        help="Qdrant service URL.",
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Qdrant collection name.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="SentenceTransformer model name.",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=DEFAULT_DEVICE,
        help="Device used for query embedding.",
    )

    return parser.parse_args()


def main() -> int:
    """Run dense retrieval from the command line."""

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
    )

    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error(
            "CUDA was requested but is unavailable. "
            "Use --device cpu or check the CUDA environment."
        )
        return 1

    retriever: DenseRetriever | None = None

    try:
        retriever = DenseRetriever(
            qdrant_url=args.qdrant_url,
            collection_name=args.collection,
            model_name=args.model,
            device=args.device,
        )

        results = retriever.search(
            query=args.query,
            top_k=args.top_k,
            arxiv_id=args.arxiv_id,
        )

        print("\n" + "=" * 90)
        print("DENSE RETRIEVAL RESULTS")
        print("=" * 90)
        print(f"Query: {args.query}")
        print(f"Collection: {args.collection}")
        print(f"Results: {len(results)}")

        if args.arxiv_id:
            print(f"arXiv filter: {args.arxiv_id}")

        print("-" * 90)

        if not results:
            print("No matching passages found.")

        for result in results:
            excerpt = result["text"].replace("\n", " ").strip()

            if len(excerpt) > 600:
                excerpt = excerpt[:600].rstrip() + "..."

            print(f"\nRank: {result['rank']}")
            print(f"Cosine similarity: {result['score']:.6f}")
            print(f"Title: {result['title']}")
            print(f"arXiv ID: {result['arxiv_id']}")
            print(f"Page: {result['page_number']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Text: {excerpt}")
            print("-" * 90)

        return 0

    except Exception:
        logger.exception("Dense retrieval failed.")
        return 1

    finally:
        if retriever is not None:
            retriever.close()


if __name__ == "__main__":
    sys.exit(main())
