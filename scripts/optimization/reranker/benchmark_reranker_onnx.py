
"""Quantize the FP32 ONNX reranker and benchmark FP32 vs INT8 inference.

Uses the exact query and candidate passages from the saved PyTorch baseline.
Measures tokenization plus model inference, and compares scores and rankings.
The production reranker and original baseline artifact are never modified.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

# Set CPU thread limits before importing numerical and ML libraries.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer


LOGGER = logging.getLogger("reranker_onnx_benchmark")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASELINE_PATH = PROJECT_ROOT / "artifacts/benchmarks/reranker_baseline.json"
FP32_PATH = PROJECT_ROOT / "artifacts/models/bge-reranker-v2-m3-fp32.onnx"
INT8_PATH = PROJECT_ROOT / "artifacts/models/bge-reranker-v2-m3-int8.onnx"
REPORT_PATH = PROJECT_ROOT / "artifacts/benchmarks/reranker_onnx_comparison.json"

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

MAX_LENGTH = 1024
BATCH_SIZE = 8
WARMUP_RUNS = 2
MEASURED_RUNS = 5

INTRA_OP_THREADS = 8
INTER_OP_THREADS = 4

# The PyTorch reference latency is from the thread-matched baseline run.
PYTORCH_BASELINE_MEDIAN_SECONDS = 10.704


def load_baseline() -> tuple[str, list[dict[str, Any]], dict[str, float]]:
    """Load the original query, passages, and PyTorch reference scores."""
    if not BASELINE_PATH.is_file():
        raise FileNotFoundError(f"Missing baseline: {BASELINE_PATH}")

    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    candidates = data["candidates"]
    reference_scores = {
        item["chunk_id"]: float(item["reranker_score"])
        for item in data["ranking"]
    }

    if not candidates:
        raise ValueError("The baseline contains no candidate passages.")

    if any(item["chunk_id"] not in reference_scores for item in candidates):
        raise ValueError("Some baseline candidates have no reference score.")

    return data["query"], candidates, reference_scores


def create_session(model_path: Path) -> ort.InferenceSession:
    """Create a deterministic CPU-only ONNX Runtime inference session."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = INTRA_OP_THREADS
    options.inter_op_num_threads = INTER_OP_THREADS
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    if session.get_providers()[0] != "CPUExecutionProvider":
        raise RuntimeError(
            f"Expected CPUExecutionProvider, got {session.get_providers()}"
        )

    return session


def tokenize_batch(
    tokenizer: Any,
    query: str,
    passages: list[str],
) -> dict[str, np.ndarray]:
    """Tokenize query–passage pairs using the original reranker settings."""
    encoded = tokenizer(
        [query] * len(passages),
        passages,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np",
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def infer_all_candidates(
    session: ort.InferenceSession,
    tokenizer: Any,
    query: str,
    candidates: list[dict[str, Any]],
) -> np.ndarray:
    """Score every candidate in the same batches used by the baseline."""
    all_scores: list[np.ndarray] = []

    for start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[start : start + BATCH_SIZE]
        passages = [item["text"] for item in batch]

        encoded = tokenize_batch(tokenizer, query, passages)

        logits = session.run(
            ["logits"],
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
        )[0]

        # The original CrossEncoder uses sigmoid activation for this model.
        # Apply it here so the ONNX outputs match the original score scale.
        scores = 1.0 / (1.0 + np.exp(-logits))
        all_scores.append(scores.squeeze(-1).astype(np.float32))

    return np.concatenate(all_scores)


def measure_latency(
    session: ort.InferenceSession,
    tokenizer: Any,
    query: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Warm up the session and measure repeated end-to-end scoring runs."""
    LOGGER.info("Running %d warm-up iterations...", WARMUP_RUNS)

    for _ in range(WARMUP_RUNS):
        infer_all_candidates(session, tokenizer, query, candidates)

    durations: list[float] = []

    for run_number in range(1, MEASURED_RUNS + 1):
        start = time.perf_counter()
        infer_all_candidates(session, tokenizer, query, candidates)
        elapsed = time.perf_counter() - start

        durations.append(elapsed)
        LOGGER.info("Measured run %d/%d: %.3f seconds",
                    run_number, MEASURED_RUNS, elapsed)

    return {
        "durations_seconds": durations,
        "median_seconds": float(np.median(durations)),
        "mean_seconds": float(np.mean(durations)),
        "min_seconds": float(np.min(durations)),
        "max_seconds": float(np.max(durations)),
    }


def ranking_ids(
    scores: np.ndarray,
    candidates: list[dict[str, Any]],
) -> list[str]:
    """Return candidate IDs ordered by score, with a deterministic tie-break."""
    ordered = sorted(
        zip(candidates, scores),
        key=lambda pair: (-float(pair[1]), pair[0]["chunk_id"]),
    )
    return [candidate["chunk_id"] for candidate, _ in ordered]


def compare_scores(
    scores: np.ndarray,
    candidates: list[dict[str, Any]],
    reference_scores: dict[str, float],
) -> dict[str, Any]:
    """Compare model scores and ranking against the saved PyTorch baseline."""
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
    reference_ids = [item["chunk_id"] for item in reference_order]
    current_ids = ranking_ids(scores, candidates)

    reference_top5 = reference_ids[:5]
    current_top5 = current_ids[:5]

    return {
        "max_absolute_score_error": float(errors.max()),
        "mean_absolute_score_error": float(errors.mean()),
        "top5_exact_match": current_top5 == reference_top5,
        "top5_overlap_count": len(set(current_top5) & set(reference_top5)),
        "full_ranking_exact_match": current_ids == reference_ids,
        "reference_top5": reference_top5,
        "current_top5": current_top5,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark FP32 and dynamic INT8 ONNX rerankers."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing INT8 model and comparison report.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not FP32_PATH.is_file():
        raise FileNotFoundError(
            f"FP32 ONNX model not found: {FP32_PATH}. "
            "Run export_reranker_onnx.py first."
        )

    if not args.overwrite and (INT8_PATH.exists() or REPORT_PATH.exists()):
        raise FileExistsError(
            "An INT8 model or comparison report already exists. "
            "Inspect it first, or rerun with --overwrite."
        )

    INT8_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    query, candidates, reference_scores = load_baseline()

    LOGGER.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    LOGGER.info("Quantizing FP32 ONNX weights to dynamic INT8...")
    quantize_dynamic(
        model_input=str(FP32_PATH),
        model_output=str(INT8_PATH),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    LOGGER.info("Creating FP32 and INT8 CPU sessions...")
    fp32_session = create_session(FP32_PATH)
    int8_session = create_session(INT8_PATH)

    LOGGER.info("Checking FP32 ONNX scores against the saved baseline...")
    fp32_scores = infer_all_candidates(
        fp32_session, tokenizer, query, candidates
    )
    fp32_comparison = compare_scores(
        fp32_scores, candidates, reference_scores
    )

    LOGGER.info("Checking INT8 ONNX scores against the saved baseline...")
    int8_scores = infer_all_candidates(
        int8_session, tokenizer, query, candidates
    )
    int8_comparison = compare_scores(
        int8_scores, candidates, reference_scores
    )

    LOGGER.info("Benchmarking FP32 ONNX...")
    fp32_timing = measure_latency(
        fp32_session, tokenizer, query, candidates
    )

    LOGGER.info("Benchmarking INT8 ONNX...")
    int8_timing = measure_latency(
        int8_session, tokenizer, query, candidates
    )

    fp32_median = fp32_timing["median_seconds"]
    int8_median = int8_timing["median_seconds"]

    report = {
        "model": MODEL_NAME,
        "candidate_count": len(candidates),
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "thread_configuration": {
            "intra_op": INTRA_OP_THREADS,
            "inter_op": INTER_OP_THREADS,
        },
        "quantization": {
            "method": "dynamic",
            "weight_type": "QInt8",
            "per_channel": True,
            "operators": ["MatMul", "Gemm"],
        },
        "artifacts": {
            "fp32_model": str(FP32_PATH.relative_to(PROJECT_ROOT)),
            "int8_model": str(INT8_PATH.relative_to(PROJECT_ROOT)),
            "fp32_size_mib": round(FP32_PATH.stat().st_size / (1024 ** 2), 2),
            "int8_size_mib": round(INT8_PATH.stat().st_size / (1024 ** 2), 2),
        },
        "pytorch_baseline": {
            "median_seconds": PYTORCH_BASELINE_MEDIAN_SECONDS,
        },
        "fp32_onnx": {
            "timing": fp32_timing,
            "quality": fp32_comparison,
        },
        "int8_onnx": {
            "timing": int8_timing,
            "quality": int8_comparison,
        },
        "speedup": {
            "int8_vs_fp32": (
                fp32_median / int8_median if int8_median > 0 else None
            ),
            "int8_vs_pytorch_baseline": (
                PYTORCH_BASELINE_MEDIAN_SECONDS / int8_median
                if int8_median > 0
                else None
            ),
            "fp32_vs_pytorch_baseline": (
                PYTORCH_BASELINE_MEDIAN_SECONDS / fp32_median
                if fp32_median > 0
                else None
            ),
        },
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("========== BENCHMARK SUMMARY ==========")
    LOGGER.info("FP32 ONNX median: %.3f seconds", fp32_median)
    LOGGER.info("INT8 ONNX median: %.3f seconds", int8_median)
    LOGGER.info(
        "INT8 speedup over FP32: %.2fx",
        report["speedup"]["int8_vs_fp32"],
    )
    LOGGER.info(
        "INT8 speedup over PyTorch baseline: %.2fx",
        report["speedup"]["int8_vs_pytorch_baseline"],
    )
    LOGGER.info(
        "INT8 top-5 exact match: %s",
        int8_comparison["top5_exact_match"],
    )
    LOGGER.info(
        "INT8 top-5 overlap: %d/5",
        int8_comparison["top5_overlap_count"],
    )
    LOGGER.info(
        "INT8 maximum score error: %.8g",
        int8_comparison["max_absolute_score_error"],
    )
    LOGGER.info("Report saved to %s", REPORT_PATH)


if __name__ == "__main__":
    main()
