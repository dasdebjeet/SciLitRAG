
"""
Cross-encoder reranking for SciLitRAG.

This module improves the ordering of hybrid retrieval candidates by evaluating
each query-passage pair jointly with a cross-encoder model.

Responsibilities:
    - Initialize a pretrained cross-encoder reranking model.
    - Accept candidate passages from the hybrid retrieval pipeline.
    - Calculate query-passage relevance scores in batches.
    - Reorder candidates using cross-encoder scores.
    - Preserve all original retrieval scores and metadata.

Assumptions:
    - Candidate passages contain a non-empty "text" field.
    - Hybrid retrieval results contain a unique "chunk_id".
    - The selected model is compatible with SentenceTransformers CrossEncoder.

Limitations:
    - Reranking only considers the candidates supplied to it.
    - Cross-encoder scores are relevance scores, not probabilities.
    - Model inference can require significant GPU memory.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from typing import Any

import torch
from sentence_transformers import CrossEncoder

from hybrid_retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 20
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 1024
DEFAULT_DEVICE = "cuda"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-Encoder Reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Score and reorder retrieval candidates using a cross-encoder.

    Unlike bi-encoder retrieval, the cross-encoder processes the query and
    passage together, allowing the model to examine their interaction.

    Args:
        model_name: Hugging Face cross-encoder model identifier.
        device: Inference device, either CUDA or CPU.
        batch_size: Number of query-passage pairs processed per batch.
        max_length: Maximum combined token sequence length.

    Raises:
        ValueError: If initialization parameters are invalid.
        RuntimeError: If the model cannot be loaded.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:

        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero.")

        if max_length < 1:
            raise ValueError("max_length must be greater than zero.")

        if device not in {"cuda", "cpu"}:
            raise ValueError("device must be either 'cuda' or 'cpu'.")

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        self.batch_size = batch_size
        self.device = device

        LOGGER.info(
            "Loading cross-encoder model: %s",
            model_name,
        )

        try:
            self.model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=device,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load reranker model '{model_name}': {exc}"
            ) from exc

        LOGGER.info(
            "Cross-encoder loaded | device=%s | batch_size=%d | max_length=%d",
            device,
            batch_size,
            max_length,
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """Rerank candidate passages using query-passage relevance scores.

        Args:
            query: Original user search query.
            candidates: Ranked candidates from hybrid retrieval.
            top_k: Maximum number of reranked passages to return.

        Returns:
            Candidates sorted by descending cross-encoder score.
            Original retrieval metadata and scores are preserved.

        Raises:
            ValueError: If the query or parameters are invalid.
            RuntimeError: If scoring fails or produces invalid results.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        if not candidates:
            LOGGER.warning("No candidates received for reranking.")
            return []

        if top_k > len(candidates):
            LOGGER.warning(
                "Requested top_k=%d but only %d candidates are available.",
                top_k,
                len(candidates),
            )

        # Build query-passage pairs in the exact order of the candidate list.
        sentence_pairs: list[tuple[str, str]] = []

        for candidate in candidates:
            text = candidate.get("text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Candidate contains missing or empty passage text."
                )

            sentence_pairs.append((query, text))

        LOGGER.info(
            "Scoring %d query-passage pairs.",
            len(sentence_pairs),
        )

        try:
            # CrossEncoder returns one relevance score for each pair.
            # These scores are used for relative ranking, not as probabilities.
            scores = self.model.predict(
                sentence_pairs,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Cross-encoder inference failed: {exc}"
            ) from exc

        # Normalize the returned score array to a flat Python list.
        score_values = scores.reshape(-1).tolist()

        if len(score_values) != len(candidates):
            raise RuntimeError(
                "Cross-encoder returned an unexpected number of scores."
            )

        reranked_results: list[dict[str, Any]] = []

        for candidate, score in zip(candidates, score_values):
            relevance_score = float(score)

            if not math.isfinite(relevance_score):
                raise RuntimeError(
                    "Cross-encoder returned a non-finite relevance score."
                )

            # Preserve the original RRF rank before assigning the new rank.
            result = dict(candidate)
            result["rrf_rank"] = candidate.get("rank")
            result["reranker_score"] = relevance_score

            reranked_results.append(result)

        # Stable tie-breaking retains deterministic output for equal scores.
        reranked_results.sort(
            key=lambda item: (
                -item["reranker_score"],
                item["chunk_id"],
            )
        )

        final_results = reranked_results[:top_k]

        # Replace the displayed rank with the new cross-encoder rank.
        for rank, result in enumerate(final_results, start=1):
            result["rank"] = rank

        LOGGER.info(
            "Reranking completed | candidates=%d | returned=%d",
            len(candidates),
            len(final_results),
        )

        return final_results


# ---------------------------------------------------------------------------
# Command-Line Interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for hybrid retrieval and reranking."""

    parser = argparse.ArgumentParser(
        description=(
            "Retrieve research passages using BM25 and dense search, "
            "combine them with RRF, and rerank using a cross-encoder."
        )
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
        help="Number of final reranked passages.",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Number of fused RRF candidates passed to the reranker.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF smoothing constant.",
    )

    parser.add_argument(
        "--arxiv-id",
        type=str,
        default=None,
        help="Optional paper-level arXiv ID filter.",
    )

    parser.add_argument(
        "--reranker-model",
        type=str,
        default=DEFAULT_RERANKER_MODEL,
        help="Hugging Face cross-encoder model identifier.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Cross-encoder inference batch size.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum combined query-passage token length.",
    )

    parser.add_argument(
        "--chunks",
        type=str,
        default=None,
        help="Optional source chunks JSONL path.",
    )

    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:16333",
        help="Qdrant service URL.",
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="scilit_rag",
        help="Qdrant collection name.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-m3",
        help="Dense embedding model identifier.",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=DEFAULT_DEVICE,
        help="Device used by both retrieval and reranking models.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_results(
    query: str,
    results: list[dict[str, Any]],
    candidate_k: int,
    args: argparse.Namespace,
) -> None:
    """Print reranked passages with their original retrieval information."""

    print("\n" + "=" * 110)
    print("SCILITRAG — HYBRID RETRIEVAL + CROSS-ENCODER RERANKING")
    print("=" * 110)

    print(f"Query: {query}")
    print(f"RRF candidates: {candidate_k}")
    print(f"Reranker model: {args.reranker_model}")
    print(f"Final results: {len(results)}")

    if args.arxiv_id:
        print(f"arXiv filter: {args.arxiv_id}")

    print("-" * 110)

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Previous RRF rank: {result['rrf_rank']}")
        print(f"Cross-encoder score: {result['reranker_score']:.6f}")
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
        print(f"Text: {result['text'][:700]}...")


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    """Execute hybrid retrieval followed by cross-encoder reranking."""

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
    )

    args = parse_args()

    if args.top_k < 1 or args.candidate_k < 1:
        LOGGER.error("top-k and candidate-k must be greater than zero.")
        return 1

    if args.candidate_k < args.top_k:
        LOGGER.error(
            "candidate-k must be greater than or equal to top-k."
        )
        return 1

    if args.device == "cuda" and not torch.cuda.is_available():
        LOGGER.error("CUDA was requested but is unavailable.")
        return 1

    hybrid: HybridRetriever | None = None

    try:
        LOGGER.info("Initializing SciLitRAG reranking pipeline.")

        hybrid_kwargs: dict[str, Any] = {
            "qdrant_url": args.qdrant_url,
            "collection_name": args.collection,
            "model_name": args.model,
            "device": args.device,
        }

        if args.chunks is not None:
            from pathlib import Path

            hybrid_kwargs["chunks_path"] = Path(args.chunks)

        # Reuse the existing retrieval implementation without duplicating
        # BM25 indexing, dense search, metadata filtering, or RRF logic.
        hybrid = HybridRetriever(**hybrid_kwargs)

        # Request candidate_k fused results rather than the final top_k.
        # The cross-encoder needs the larger candidate pool to improve ranking.
        candidates = hybrid.search(
            query=args.query,
            top_k=args.candidate_k,
            retrieval_k=args.candidate_k,
            rrf_k=args.rrf_k,
            arxiv_id=args.arxiv_id,
        )

        LOGGER.info(
            "RRF candidate pool ready | candidates=%d",
            len(candidates),
        )

        reranker = CrossEncoderReranker(
            model_name=args.reranker_model,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        results = reranker.rerank(
            query=args.query,
            candidates=candidates,
            top_k=args.top_k,
        )

        display_results(
            query=args.query,
            results=results,
            candidate_k=len(candidates),
            args=args,
        )

        LOGGER.info("SciLitRAG reranking pipeline completed.")

        return 0

    except KeyboardInterrupt:
        LOGGER.warning("Execution interrupted by user.")
        return 130

    except Exception:
        LOGGER.exception("Reranking pipeline failed.")
        return 1

    finally:
        if hybrid is not None:
            hybrid.close()


if __name__ == "__main__":
    sys.exit(main())
