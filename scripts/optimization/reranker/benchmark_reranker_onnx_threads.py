
"""Find the best intra-op thread count for FP32 ONNX reranking.

Uses the exact query and candidates from the saved PyTorch baseline.
Measures inference-only latency for sequential ONNX Runtime execution.

This is an isolated benchmark: it does not modify production code,
the original baseline, or existing ONNX model artifacts.
"""

from __future__ import annotations

import gc
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

# Set native CPU thread limits before importing numerical libraries.
# ONNX Runtime's intra-op thread count is configured separately per session.
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


LOGGER = logging.getLogger("reranker_onnx_thread_benchmark")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASELINE_PATH = (
    PROJECT_ROOT / "artifacts/benchmarks/reranker_baseline.json"
)
MODEL_PATH = (
    PROJECT_ROOT / "artifacts/models/bge-reranker-v2-m3-fp32.onnx"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts/benchmarks/reranker_onnx_thread_comparison.json"
)

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Keep the model input identical to the existing benchmark.
BATCH_SIZE = 8
MAX_LENGTH = 1024

# A short benchmark is sufficient to identify a promising configuration.
WARMUP_RUNS = 1
MEASURED_RUNS = 3

# Test the available CPU parallelism without changing model precision.
THREAD_COUNTS = [1, 2, 4, 8, 16]

# Sequential graph execution avoids the parallel scheduling overhead
# observed in the previous experiment.
INTER_OP_THREADS = 1

# FP32 ONNX should closely reproduce the saved PyTorch scores.
MAX_ALLOWED_SCORE_ERROR = 1e-5


def load_baseline() -> tuple[
    str,
    list[dict[str, Any]],
    dict[str, float],
]:
    """Load the fixed query, candidates, and PyTorch reference scores."""

    if not BASELINE_PATH.is_file():
        raise FileNotFoundError(f"Missing baseline: {BASELINE_PATH}")

    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    candidates = data["candidates"]
    reference_scores = {
        item["chunk_id"]: float(item["reranker_score"])
        for item in data["ranking"]
    }

    if not candidates:
        raise ValueError("Baseline contains no candidates.")

    missing_ids = [
        item["chunk_id"]
        for item in candidates
        if item["chunk_id"] not in reference_scores
    ]
    if missing_ids:
        raise ValueError(
            f"Candidates missing reference scores: {missing_ids}"
        )

    return data["query"], candidates, reference_scores


def tokenize_candidates(
    tokenizer: Any,
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, np.ndarray]]:
    """Tokenize all candidate pairs once for fair inference comparisons."""

    batches = []

    for start in range(0, len(candidates), BATCH_SIZE):
        candidate_batch = candidates[start : start + BATCH_SIZE]
        passages = [str(item["text"]) for item in candidate_batch]

        encoded = tokenizer(
            [query] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )

        batches.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        })

    return batches


def create_session(
    intra_op_threads: int,
) -> ort.InferenceSession:
    """Create a CPU-only sequential session with the requested thread count."""

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = INTER_OP_THREADS
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # Avoid spinning idle worker threads between inference calls.
    options.add_session_config_entry(
        "session.intra_op.allow_spinning", "0"
    )
    options.add_session_config_entry(
        "session.inter_op.allow_spinning", "0"
    )

    return ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def run_inference(
    session: ort.InferenceSession,
    batches: list[dict[str, np.ndarray]],
) -> np.ndarray:
    """Run all batches and return one sigmoid score per candidate."""

    output_name = session.get_outputs()[0].name
    input_names = {item.name for item in session.get_inputs()}

    required_inputs = {"input_ids", "attention_mask"}
    if not required_inputs.issubset(input_names):
        raise RuntimeError(
            f"Unexpected ONNX inputs: {sorted(input_names)}"
        )

    all_scores = []

    for batch in batches:
        logits = session.run(
            [output_name],
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            },
        )[0]

        # Match the CrossEncoder's default sigmoid activation.
        scores = 1.0 / (1.0 + np.exp(-logits))
        all_scores.append(np.asarray(scores).reshape(-1))

    return np.concatenate(all_scores).astype(np.float32)


def evaluate_quality(
    scores: np.ndarray,
    candidates: list[dict[str, Any]],
    reference_scores: dict[str, float],
) -> dict[str, Any]:
    """Compare scores and rankings with the saved PyTorch baseline."""

    reference = np.asarray(
        [reference_scores[item["chunk_id"]] for item in candidates],
        dtype=np.float32,
    )

    if scores.shape != reference.shape:
        raise ValueError(
            f"Score shape mismatch: ONNX={scores.shape}, "
            f"reference={reference.shape}"
        )

    errors = np.abs(scores - reference)

    # Use the same deterministic tie-breaking rule as the baseline.
    reference_order = sorted(
        candidates,
        key=lambda item: (
            -reference_scores[item["chunk_id"]],
            item["chunk_id"],
        ),
    )

    current_order = sorted(
        zip(candidates, scores),
        key=lambda pair: (-float(pair[1]), pair[0]["chunk_id"]),
    )

    reference_ids = [item["chunk_id"] for item in reference_order]
    current_ids = [
        candidate["chunk_id"] for candidate, _ in current_order
    ]

    return {
        "max_absolute_score_error": float(errors.max()),
        "mean_absolute_score_error": float(errors.mean()),
        "top5_exact_match": current_ids[:5] == reference_ids[:5],
        "full_ranking_exact_match": current_ids == reference_ids,
        "top5_overlap_count": len(
            set(current_ids[:5]) & set(reference_ids[:5])
        ),
        "reference_top5": reference_ids[:5],
        "onnx_top5": current_ids[:5],
        "within_score_tolerance": (
            float(errors.max()) <= MAX_ALLOWED_SCORE_ERROR
        ),
    }


def benchmark_configuration(
    thread_count: int,
    batches: list[dict[str, np.ndarray]],
    candidates: list[dict[str, Any]],
    reference_scores: dict[str, float],
) -> dict[str, Any]:
    """Measure one thread configuration and release its session afterward."""

    LOGGER.info(
        "Creating sequential session with %d intra-op threads...",
        thread_count,
    )

    session = create_session(thread_count)

    try:
        # Warm up model execution before collecting measurements.
        for _ in range(WARMUP_RUNS):
            run_inference(session, batches)

        # Validate numerical equivalence independently of timing.
        scores = run_inference(session, batches)
        quality = evaluate_quality(
            scores, candidates, reference_scores
        )

        durations = []

        for run_number in range(1, MEASURED_RUNS + 1):
            start = time.perf_counter()
            run_inference(session, batches)
            elapsed = time.perf_counter() - start

            durations.append(elapsed)

            LOGGER.info(
                "%d threads | run %d/%d: %.4f seconds",
                thread_count,
                run_number,
                MEASURED_RUNS,
                elapsed,
            )

        return {
            "intra_op_threads": thread_count,
            "execution_mode": "ORT_SEQUENTIAL",
            "inter_op_threads": INTER_OP_THREADS,
            "durations_seconds": durations,
            "median_seconds": float(statistics.median(durations)),
            "mean_seconds": float(statistics.mean(durations)),
            "minimum_seconds": float(min(durations)),
            "maximum_seconds": float(max(durations)),
            "quality": quality,
        }

    finally:
        # Only one benchmark session should remain active at a time.
        del session
        gc.collect()


def main() -> None:
    """Run the thread sweep and save a separate benchmark report."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing ONNX model: {MODEL_PATH}")

    if REPORT_PATH.exists():
        raise FileExistsError(
            f"Report already exists: {REPORT_PATH}. "
            "Rename it before running another experiment."
        )

    query, candidates, reference_scores = load_baseline()

    LOGGER.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    LOGGER.info("Tokenizing %d candidates once...", len(candidates))
    batches = tokenize_candidates(tokenizer, query, candidates)

    results = []

    for thread_count in THREAD_COUNTS:
        LOGGER.info(
            "========== INTRA-OP THREADS: %d ==========",
            thread_count,
        )

        result = benchmark_configuration(
            thread_count,
            batches,
            candidates,
            reference_scores,
        )
        results.append(result)

        LOGGER.info(
            "%d threads | median %.3fs | top-5 exact: %s | "
            "max score error: %.8g",
            thread_count,
            result["median_seconds"],
            result["quality"]["top5_exact_match"],
            result["quality"]["max_absolute_score_error"],
        )

    # A configuration is eligible only when its scores and ranking
    # reproduce the saved PyTorch baseline within the chosen tolerance.
    eligible = [
        item for item in results
        if item["quality"]["within_score_tolerance"]
        and item["quality"]["top5_exact_match"]
    ]

    best = min(
        eligible,
        key=lambda item: item["median_seconds"],
    ) if eligible else None

    report = {
        "model": MODEL_NAME,
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "candidate_count": len(candidates),
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "execution_mode": "ORT_SEQUENTIAL",
        "graph_optimization": "ORT_ENABLE_ALL",
        "inter_op_threads": INTER_OP_THREADS,
        "tested_intra_op_threads": THREAD_COUNTS,
        "score_error_tolerance": MAX_ALLOWED_SCORE_ERROR,
        "results": results,
        "best_eligible_configuration": best,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("========== FINAL THREAD SWEEP ==========")

    for item in results:
        LOGGER.info(
            "%2d threads | median %.3fs | min %.3fs | max %.3fs | "
            "top-5 exact: %s | score tolerance: %s",
            item["intra_op_threads"],
            item["median_seconds"],
            item["minimum_seconds"],
            item["maximum_seconds"],
            item["quality"]["top5_exact_match"],
            item["quality"]["within_score_tolerance"],
        )

    if best:
        LOGGER.info(
            "Best eligible configuration: %d threads, %.3fs median",
            best["intra_op_threads"],
            best["median_seconds"],
        )
    else:
        LOGGER.warning(
            "No configuration passed the numerical and ranking checks."
        )

    LOGGER.info("Report saved to: %s", REPORT_PATH)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()