
"""
Hybrid retrieval for SciLitRAG using BM25, dense search, and RRF.

This module combines two complementary retrieval methods:

    1. BM25: lexical retrieval using matching terms.
    2. Dense retrieval: semantic retrieval using BAAI/bge-m3 and Qdrant.
    3. Reciprocal Rank Fusion (RRF): combines the ranked result lists.

Responsibilities:
    - Initialize the BM25 and dense retrievers once per process.
    - Execute both retrieval methods for the same query.
    - Optionally restrict both methods to a specific arXiv paper.
    - Merge duplicate chunks using their deterministic chunk IDs.
    - Calculate RRF scores and return a unified ranked result list.

Assumptions:
    - Source chunks exist at data/processed/chunks.jsonl.
    - Qdrant is running and contains the populated scilit_rag collection.
    - The dense retriever uses the same BGE-M3 model as corpus embeddings.

Limitations:
    - RRF combines rankings, not the underlying BM25 and cosine scores.
    - RRF does not independently judge passage relevance.
    - Cross-encoder reranking is intentionally handled in a later stage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from scilitrag.core.paths import DEFAULT_CHUNKS_PATH
from scilitrag.retrieval.bm25_retriever import BM25Retriever, load_chunks
from scilitrag.retrieval.dense_retriever import DenseRetriever


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


DEFAULT_QDRANT_URL = "http://localhost:16333"
DEFAULT_COLLECTION = "scilit_rag"
DEFAULT_MODEL = "BAAI/bge-m3"

DEFAULT_TOP_K = 5
DEFAULT_RETRIEVAL_K = 20
DEFAULT_RRF_K = 60
DEFAULT_DEVICE = "cuda"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Combine lexical and semantic retrieval using Reciprocal Rank Fusion.

    Both retrieval engines are initialized once and reused across searches.
    BM25 operates over the local chunk corpus, while dense retrieval queries
    the existing Qdrant collection.

    Args:
        chunks_path: Path to the source chunks JSONL file.
        qdrant_url: URL of the Qdrant service.
        collection_name: Existing Qdrant collection.
        model_name: SentenceTransformer embedding model.
        device: Device used for query embedding.

    Raises:
        RuntimeError: If either retrieval engine cannot be initialized.
    """

    def __init__(
        self,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        LOGGER.info("Initializing hybrid retrieval pipeline.")

        self.bm25: BM25Retriever | None = None
        self.dense: DenseRetriever | None = None

        try:
            # Load the source corpus once and reuse it for BM25 indexing.
            chunks = load_chunks(chunks_path)

            LOGGER.info(
                "Initializing BM25 with %d source chunks.",
                len(chunks),
            )

            self.bm25 = BM25Retriever(chunks)

            # DenseRetriever loads BGE-M3 and connects to existing Qdrant.
            self.dense = DenseRetriever(
                qdrant_url=qdrant_url,
                collection_name=collection_name,
                model_name=model_name,
                device=device,
            )

            LOGGER.info("Hybrid retrieval pipeline initialized.")

        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"Failed to initialize hybrid retrieval: {exc}"
            ) from exc

    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: Sequence[list[dict[str, Any]]],
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[dict[str, Any]]:
        """Fuse multiple ranked result lists using reciprocal rank fusion.

        Each unique chunk receives a contribution from every retrieval list
        in which it appears. The contribution depends on its rank, not its
        original retrieval score.

        Args:
            result_lists: Ranked results from independent retrieval systems.
            rrf_k: Positive smoothing constant used in the RRF formula.

        Returns:
            Unique chunks sorted by descending RRF score.

        Raises:
            ValueError: If the smoothing constant is not positive.
        """

        if rrf_k < 1:
            raise ValueError("rrf_k must be greater than zero.")

        fused_results: dict[str, dict[str, Any]] = {}

        for source_index, results in enumerate(result_lists):
            source_name = "bm25" if source_index == 0 else "dense"

            for result in results:
                chunk_id = result.get("chunk_id")
                rank = result.get("rank")

                if not chunk_id or not isinstance(rank, int) or rank < 1:
                    LOGGER.warning(
                        "Skipping invalid retrieval result: %s",
                        result,
                    )
                    continue

                # A chunk retrieved by both systems should appear only once
                # in the fused output, accumulating both rank contributions.
                if chunk_id not in fused_results:
                    fused_results[chunk_id] = {
                        **result,
                        "rrf_score": 0.0,
                        "retrieval_sources": [],
                    }

                fused = fused_results[chunk_id]

                fused["rrf_score"] += 1.0 / (rrf_k + rank)
                fused["retrieval_sources"].append(source_name)

                # Preserve original scores and ranks for later analysis.
                fused[f"{source_name}_rank"] = rank
                fused[f"{source_name}_score"] = float(result["score"])

        # Stable tie-breaking makes repeated runs easier to compare.
        return sorted(
            fused_results.values(),
            key=lambda item: (
                -item["rrf_score"],
                item["chunk_id"],
            ),
        )

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        rrf_k: int = DEFAULT_RRF_K,
        arxiv_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute BM25 and dense retrieval, then fuse their rankings.

        Args:
            query: Natural-language or keyword-based search query.
            top_k: Number of final fused passages to return.
            retrieval_k: Candidate depth requested from each retriever.
            rrf_k: RRF smoothing constant.
            arxiv_id: Optional paper-level metadata filter.

        Returns:
            Fused and ranked passage dictionaries.

        Raises:
            ValueError: If query or retrieval parameters are invalid.
            RuntimeError: If either retrieval operation fails.
        """

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        if retrieval_k < top_k:
            raise ValueError(
                "retrieval_k must be greater than or equal to top_k."
            )

        if rrf_k < 1:
            raise ValueError("rrf_k must be greater than zero.")

        if self.bm25 is None or self.dense is None:
            raise RuntimeError("Hybrid retriever is not initialized.")

        LOGGER.info(
            "Starting hybrid search | candidate_depth=%d | final_top_k=%d",
            retrieval_k,
            top_k,
        )

        try:
            # Both methods receive the same query and paper filter.
            bm25_results = self.bm25.search(
                query=query,
                top_k=retrieval_k,
                arxiv_id=arxiv_id,
            )

            dense_results = self.dense.search(
                query=query,
                top_k=retrieval_k,
                arxiv_id=arxiv_id,
            )

            LOGGER.info(
                "Candidate retrieval completed | BM25=%d | dense=%d",
                len(bm25_results),
                len(dense_results),
            )

            fused_results = self.reciprocal_rank_fusion(
                result_lists=[bm25_results, dense_results],
                rrf_k=rrf_k,
            )

            final_results = fused_results[:top_k]

            # Assign final fused ranks without overwriting source-specific ranks.
            for rank, result in enumerate(final_results, start=1):
                result["rank"] = rank

            LOGGER.info(
                "RRF completed | unique_chunks=%d | returned=%d",
                len(fused_results),
                len(final_results),
            )

            return final_results

        except Exception as exc:
            raise RuntimeError(
                f"Hybrid search failed: {exc}"
            ) from exc

    def close(self) -> None:
        """Release resources owned by the dense retriever."""

        if self.dense is not None:
            self.dense.close()
            self.dense = None


# ---------------------------------------------------------------------------
# Command-Line Interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for hybrid retrieval."""

    parser = argparse.ArgumentParser(
        description="Search SciLitRAG using BM25 + dense retrieval + RRF."
    )

    parser.add_argument(
        "--query",
        type=str,
        default=(
            "How do machine learning methods quantify "
            "pore-space evolution over time?"
        ),
        help="Natural-language search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of final fused passages to return.",
    )

    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=DEFAULT_RETRIEVAL_K,
        help="Candidate count requested from each retrieval method.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF smoothing constant.",
    )

    parser.add_argument(
        "--arxiv-id",
        type=str,
        default=None,
        help="Optional arXiv ID filter.",
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to the source chunks JSONL file.",
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
    """Execute hybrid retrieval from the command line."""

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
    )

    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        LOGGER.error(
            "CUDA was requested but is unavailable. "
            "Use --device cpu or check the CUDA environment."
        )
        return 1

    retriever: HybridRetriever | None = None

    try:
        retriever = HybridRetriever(
            chunks_path=args.chunks,
            qdrant_url=args.qdrant_url,
            collection_name=args.collection,
            model_name=args.model,
            device=args.device,
        )

        results = retriever.search(
            query=args.query,
            top_k=args.top_k,
            retrieval_k=args.retrieval_k,
            rrf_k=args.rrf_k,
            arxiv_id=args.arxiv_id,
        )

        print("\n" + "=" * 100)
        print("SCILITRAG — HYBRID RETRIEVAL (BM25 + DENSE + RRF)")
        print("=" * 100)
        print(f"Query: {args.query}")
        print(f"Candidate depth per retriever: {args.retrieval_k}")
        print(f"RRF smoothing constant: {args.rrf_k}")
        print(f"Final results: {len(results)}")

        if args.arxiv_id:
            print(f"arXiv filter: {args.arxiv_id}")

        print("-" * 100)

        if not results:
            print("No matching passages found.")

        for result in results:
            excerpt = result["text"].replace("\n", " ").strip()

            if len(excerpt) > 600:
                excerpt = excerpt[:600].rstrip() + "..."

            print(f"\nRank: {result['rank']}")
            print(f"RRF score: {result['rrf_score']:.8f}")
            print(
                "Retrieved by: "
                + ", ".join(result["retrieval_sources"])
            )

            print(
                f"BM25 rank/score: "
                f"{result.get('bm25_rank', 'N/A')} / "
                f"{result.get('bm25_score', 'N/A')}"
            )

            print(
                f"Dense rank/score: "
                f"{result.get('dense_rank', 'N/A')} / "
                f"{result.get('dense_score', 'N/A')}"
            )

            print(f"Title: {result['title']}")
            print(f"arXiv ID: {result['arxiv_id']}")
            print(f"Page: {result['page_number']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Text: {excerpt}")
            print("-" * 100)

        return 0

    except Exception:
        LOGGER.exception("Hybrid retrieval execution failed.")
        return 1

    finally:
        if retriever is not None:
            retriever.close()


if __name__ == "__main__":
    sys.exit(main())
