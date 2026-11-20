
"""Compare sequential and parallel FP32 ONNX Runtime execution.

Uses the exact query and candidate passages saved in the PyTorch baseline.
Measures tokenization, model inference, and end-to-end scoring independently.

The original PyTorch baseline, existing ONNX models, and production
reranker are never modified.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import Any

# Set native CPU thread limits before importing numerical libraries.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


LOGGER = logging.getLogger("reranker_onnx_execution_benchmark")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASELINE_PATH = (
    PROJECT_ROOT / "artifacts/benchmarks/reranker_baseline.json"
)
MODEL_PATH = (
    PROJECT_ROOT / "artifacts/models/bge-reranker-v2-m3-fp32.onnx"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts/benchmarks/reranker_onnx_execution_comparison.json"
)

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

BATCH_SIZE = 8
MAX_LENGTH = 1024
WARMUP_RUNS = 2
MEASURED_RUNS = 5

INTRA_OP_THREADS = 8
INTER_OP_THREADS = 4

EXECUTION_MODES = {
    "sequential": ort.ExecutionMode.ORT_SEQUENTIAL,
    "parallel": ort.ExecutionMode.ORT_PARALLEL,
}


def load_baseline() -> tuple[
    str,
    list[dict[str, Any]],
    dict[str, float],
]:
    """Load the fixed query, candidate passages, and PyTorch scores."""

    if not BASELINE_PATH.is_file():
        raise FileNotFoundError(f"Missing baseline: {BASELINE_PATH}")

    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    candidates = data["candidates"]
    reference_scores = {
        item["chunk_id"]: float(item["reranker_score"])
        for item in data["ranking"]
    }

    if not candidates:
        raise ValueError("The baseline contains no candidates.")

    if any(item["chunk_id"] not in reference_scores for item in candidates):
        raise ValueError("Some candidates have no reference score.")

    return data["query"], candidates, reference_scores


def create_session(
    execution_mode: ort.ExecutionMode,
) -> ort.InferenceSession:
    """Create a CPU-only FP32 ONNX session with controlled threading."""

    options = ort.SessionOptions()
    options.intra_op_num_threads = INTRA_OP_THREADS
    options.inter_op_num_threads = INTER_OP_THREADS
    options.execution_mode = execution_mode
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # Disable spinning to reduce idle CPU consumption between operations.
    options.add_session_config_entry(
        "session.intra_op.allow_spinning", "0"
    )
    options.add_session_config_entry(
        "session.inter_op.allow_spinning", "0"
    )

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    if "CPUExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            f"CPUExecutionProvider unavailable: {session.get_providers()}"
        )

    return session


def tokenize_candidates(
    tokenizer: Any,
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, np.ndarray]]:
    """Tokenize all candidate pairs using the original CrossEncoder settings."""

    tokenized_batches = []

    for start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[start : start + BATCH_SIZE]
        passages = [str(item["text"]) for item in batch]

        encoded = tokenizer(
            [query] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )

        tokenized_batches.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        })

    return tokenized_batches


def run_inference(
    session: ort.InferenceSession,
    tokenized_batches: list[dict[str, np.ndarray]],
) -> np.ndarray:
    """Run the model on pre-tokenized batches and return sigmoid scores."""

    input_names = {item.name for item in session.get_inputs()}
    output_name = session.get_outputs()[0].name

    required_inputs = {"input_ids", "attention_mask"}

    if not required_inputs.issubset(input_names):
        raise RuntimeError(
            f"Unexpected ONNX inputs: {sorted(input_names)}"
        )

    all_scores = []

    for batch in tokenized_batches:
        outputs = session.run(
            [output_name],
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            },
        )

        logits = outputs[0]
        scores = 1.0 / (1.0 + np.exp(-logits))
        all_scores.append(np.asarray(scores).reshape(-1))

    return np.concatenate(all_scores).astype(np.float32)


def measure_stage(
    operation: Any,
    warmup_runs: int = WARMUP_RUNS,
) -> dict[str, Any]:
    """Measure one isolated stage after excluding warmup iterations."""

    for _ in range(warmup_runs):
        operation()

    durations = []

    for run_number in range(1, MEASURED_RUNS + 1):
        start = time.perf_counter()
        operation()
        elapsed = time.perf_counter() - start

        durations.append(elapsed)

        LOGGER.info(
            "Run %d/%d: %.4f seconds",
            run_number,
            MEASURED_RUNS,
            elapsed,
        )

    return {
        "durations_seconds": durations,
        "median_seconds": float(statistics.median(durations)),
        "mean_seconds": float(statistics.mean(durations)),
        "minimum_seconds": float(min(durations)),
        "maximum_seconds": float(max(durations)),
    }


def measure_end_to_end(
    session: ort.InferenceSession,
    tokenizer: Any,
    query: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure tokenization and inference together, as a real request would."""

    def operation() -> None:
        batches = tokenize_candidates(tokenizer, query, candidates)
        run_inference(session, batches)

    return measure_stage(operation)


def compare_scores(
    scores: np.ndarray,
    candidates: list[dict[str, Any]],
    reference_scores: dict[str, float],
) -> dict[str, Any]:
    """Compare ONNX scores and rankings against the PyTorch reference."""

    reference = np.asarray(
        [reference_scores[item["chunk_id"]] for item in candidates],
        dtype=np.float32,
    )

    errors = np.abs(scores - reference)

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
        "top5_overlap_count": len(
            set(current_ids[:5]) & set(reference_ids[:5])
        ),
        "full_ranking_exact_match": current_ids == reference_ids,
        "reference_top5": reference_ids[:5],
        "current_top5": current_ids[:5],
    }


def benchmark_mode(
    name: str,
    mode: ort.ExecutionMode,
    tokenizer: Any,
    query: str,
    candidates: list[dict[str, Any]],
    reference_scores: dict[str, float],
) -> dict[str, Any]:
    """Benchmark one execution mode with identical inputs and settings."""

    LOGGER.info("Creating %s execution session...", name)

    session = create_session(mode)

    try:
        LOGGER.info("Tokenizing candidate passages...")
        tokenization_timing = measure_stage(
            lambda: tokenize_candidates(tokenizer, query, candidates)
        )

        # Pre-tokenize once so inference-only timing excludes tokenizer cost.
        tokenized_batches = tokenize_candidates(
            tokenizer, query, candidates
        )

        LOGGER.info("Checking %s model scores...", name)
        scores = run_inference(session, tokenized_batches)
        quality = compare_scores(scores, candidates, reference_scores)

        LOGGER.info("Measuring %s inference-only latency...", name)
        inference_timing = measure_stage(
            lambda: run_inference(session, tokenized_batches)
        )

        LOGGER.info("Measuring %s end-to-end latency...", name)
        end_to_end_timing = measure_end_to_end(
            session, tokenizer, query, candidates
        )

        return {
            "execution_mode": name,
            "timing": {
                "tokenization": tokenization_timing,
                "inference_only": inference_timing,
                "end_to_end": end_to_end_timing,
            },
            "quality": quality,
        }

    finally:
        # Release the session before constructing the next configuration.
        del session


def main() -> None:
    """Run the controlled sequential-versus-parallel comparison."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"FP32 ONNX model not found: {MODEL_PATH}"
        )

    if REPORT_PATH.exists():
        raise FileExistsError(
            f"Report already exists: {REPORT_PATH}. "
            "Move or rename it before running another experiment."
        )

    query, candidates, reference_scores = load_baseline()

    LOGGER.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    LOGGER.info(
        "Candidates=%d | Batch=%d | Max length=%d | "
        "Intra-op=%d | Inter-op=%d",
        len(candidates),
        BATCH_SIZE,
        MAX_LENGTH,
        INTRA_OP_THREADS,
        INTER_OP_THREADS,
    )

    results = {}

    for name, mode in EXECUTION_MODES.items():
        LOGGER.info("========== %s ==========", name.upper())

        results[name] = benchmark_mode(
            name,
            mode,
            tokenizer,
            query,
            candidates,
            reference_scores,
        )

    sequential_median = (
        results["sequential"]["timing"]["end_to_end"]["median_seconds"]
    )
    parallel_median = (
        results["parallel"]["timing"]["end_to_end"]["median_seconds"]
    )

    report = {
        "model": MODEL_NAME,
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "candidate_count": len(candidates),
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "thread_configuration": {
            "intra_op": INTRA_OP_THREADS,
            "inter_op": INTER_OP_THREADS,
        },
        "graph_optimization": "ORT_ENABLE_ALL",
        "execution_provider": "CPUExecutionProvider",
        "results": results,
        "parallel_vs_sequential_end_to_end_speedup": (
            sequential_median / parallel_median
            if parallel_median > 0 else None
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("========== COMPARISON SUMMARY ==========")

    for name, result in results.items():
        timing = result["timing"]

        LOGGER.info(
            "%s | Tokenization: %.3fs | Inference: %.3fs | "
            "End-to-end: %.3fs",
            name.upper(),
            timing["tokenization"]["median_seconds"],
            timing["inference_only"]["median_seconds"],
            timing["end_to_end"]["median_seconds"],
        )

        LOGGER.info(
            "%s | Top-5 exact: %s | Maximum score error: %.8g",
            name.upper(),
            result["quality"]["top5_exact_match"],
            result["quality"]["max_absolute_score_error"],
        )

    LOGGER.info(
        "Parallel/sequential end-to-end speedup: %.3fx",
        report["parallel_vs_sequential_end_to_end_speedup"],
    )
    LOGGER.info("Report saved to: %s", REPORT_PATH)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
