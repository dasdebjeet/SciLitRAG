
"""Benchmark the existing SciLitRAG CPU cross-encoder reranker.

The script retrieves one fixed set of candidates, warms up the existing
PyTorch reranker, and measures repeated inference on identical inputs.

It saves the candidate texts and baseline ranking so a future ONNX
implementation can be evaluated against precisely the same passages.
"""


from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import Any

# Match the CPU thread limits configured by the SciLitRAG API.
# These environment variables must be set before importing PyTorch,
# NumPy, SciPy, or other libraries that initialize native thread pools.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import torch

# Apply the same PyTorch intra-op and inter-op limits as the API process.
torch.set_num_threads(8)
torch.set_num_interop_threads(4)

from scilitrag.retrieval.hybrid_retriever import HybridRetriever
from scilitrag.retrieval.reranker import CrossEncoderReranker


LOGGER = logging.getLogger("reranker_benchmark")

DEFAULT_CANDIDATE_K = 20
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 1024
DEFAULT_RUNS = 5
DEFAULT_WARMUP_RUNS = 1

DEFAULT_OUTPUT = Path("artifacts/benchmarks/reranker_baseline.json")


def read_memory_usage() -> dict[str, float | None]:
    """Read current Linux process RSS and anonymous RSS in MiB."""

    status_path = Path("/proc/self/status")

    if not status_path.exists():
        return {"rss_mib": None, "anonymous_mib": None}

    values: dict[str, float | None] = {
        "rss_mib": None,
        "anonymous_mib": None,
    }

    with status_path.open(encoding="utf-8") as status_file:
        for line in status_file:
            if line.startswith("VmRSS:"):
                values["rss_mib"] = int(line.split()[1]) / 1024
            elif line.startswith("RssAnon:"):
                values["anonymous_mib"] = int(line.split()[1]) / 1024

    return values


def benchmark(args: argparse.Namespace) -> None:
    """Retrieve fixed candidates and benchmark repeated reranker inference."""

    hybrid: HybridRetriever | None = None
    reranker: CrossEncoderReranker | None = None

    try:
        LOGGER.info(
            "CPU thread configuration | OMP=%s | OpenBLAS=%s | "
            "PyTorch intra-op=%d | inter-op=%d",
            os.environ["OMP_NUM_THREADS"],
            os.environ["OPENBLAS_NUM_THREADS"],
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
        )
        hybrid = HybridRetriever(
            qdrant_url="http://localhost:16333",
            collection_name="scilit_rag",
            model_name="BAAI/bge-m3",
            device="cpu",
            chunks_path=Path("data/processed/chunks.jsonl"),
        )

        LOGGER.info("Retrieving %d candidates.", args.candidate_k)

        candidates = hybrid.search(
            query=args.query,
            top_k=args.candidate_k,
            retrieval_k=args.candidate_k,
            arxiv_id=None,
        )

        if not candidates:
            raise RuntimeError(
                "No retrieval candidates were found. "
                "Check Qdrant and the indexed collection."
            )

        LOGGER.info("Retrieved %d candidates.", len(candidates))

        # Save the original candidate data before reranking. This preserves
        # the exact passages needed for a fair future ONNX comparison.
        saved_candidates = [
            {
                **candidate,
                "text": str(candidate["text"]),
            }
            for candidate in candidates
        ]

        LOGGER.info("Initializing existing PyTorch cross-encoder.")

        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-v2-m3",
            device="cpu",
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        memory_after_loading = read_memory_usage()

        # Warm-up runs are deliberately excluded from the reported latency.
        for warmup_index in range(args.warmup_runs):
            LOGGER.info(
                "Warm-up run %d/%d.",
                warmup_index + 1,
                args.warmup_runs,
            )
            reranker.rerank(
                query=args.query,
                candidates=candidates,
                top_k=len(candidates),
            )

        durations: list[float] = []
        final_ranking: list[dict[str, Any]] = []

        for run_index in range(args.runs):
            start = time.perf_counter()

            ranked = reranker.rerank(
                query=args.query,
                candidates=candidates,
                top_k=len(candidates),
            )

            elapsed = time.perf_counter() - start
            durations.append(elapsed)
            final_ranking = ranked

            LOGGER.info(
                "Run %d/%d: %.3f seconds",
                run_index + 1,
                args.runs,
                elapsed,
            )

        memory_after_inference = read_memory_usage()

        output = {
            "query": args.query,
            "model": "BAAI/bge-reranker-v2-m3",
            "backend": "sentence-transformers-pytorch",
            "device": "cpu",
            "candidate_count": len(candidates),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.runs,
            "durations_seconds": durations,
            "statistics": {
                "minimum": min(durations),
                "maximum": max(durations),
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
            },
            "memory_after_model_loading_mib": memory_after_loading,
            "memory_after_inference_mib": memory_after_inference,
            "candidates": saved_candidates,
            "ranking": [
                {
                    "chunk_id": item.get("chunk_id"),
                    "rrf_rank": item.get("rrf_rank"),
                    "reranker_score": float(item["reranker_score"]),
                }
                for item in final_ranking
            ],
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)

        with args.output.open("w", encoding="utf-8") as output_file:
            json.dump(output, output_file, indent=2, ensure_ascii=False)

        print("\n" + "=" * 55)
        print("PYTORCH RERANKER BASELINE")
        print("=" * 55)
        print(f"Candidates: {len(candidates)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Maximum length: {args.max_length}")
        print(f"Runs: {args.runs}")
        print(f"Median latency: {statistics.median(durations):.3f}s")
        print(f"Mean latency: {statistics.mean(durations):.3f}s")
        print(f"Minimum latency: {min(durations):.3f}s")
        print(f"Maximum latency: {max(durations):.3f}s")
        print(f"Results saved to: {args.output}")

    finally:
        # Release both CPU models and the Qdrant connection even if a
        # benchmark run or result serialization fails.
        if reranker is not None:
            reranker.close()

        if hybrid is not None:
            hybrid.close()


def parse_args() -> argparse.Namespace:
    """Parse reproducible benchmark settings."""

    parser = argparse.ArgumentParser(
        description="Benchmark the existing SciLitRAG PyTorch reranker."
    )

    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args()

    if args.candidate_k < 1:
        parser.error("--candidate-k must be positive.")

    if args.batch_size < 1 or args.max_length < 1:
        parser.error("--batch-size and --max-length must be positive.")

    if args.runs < 1 or args.warmup_runs < 0:
        parser.error("--runs must be positive and --warmup-runs non-negative.")

    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    benchmark(parse_args())
