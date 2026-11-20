
"""Export BGE reranker to FP32 ONNX and validate against PyTorch.

The script uses the exact query and candidate passages captured by the
PyTorch baseline benchmark. It does not modify the production reranker
or overwrite the baseline benchmark artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

# Configure CPU threading before importing PyTorch or numerical libraries.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import numpy as np
import onnx
import onnxruntime as ort
import torch
from sentence_transformers import CrossEncoder
from torch import nn


LOGGER = logging.getLogger("reranker_onnx_export")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_PATH = PROJECT_ROOT / "artifacts/benchmarks/reranker_baseline.json"
MODEL_NAME = "BAAI/bge-reranker-v2-m3"

ONNX_PATH = PROJECT_ROOT / "artifacts/models/bge-reranker-v2-m3-fp32.onnx"
REPORT_PATH = (
    PROJECT_ROOT / "artifacts/benchmarks/onnx_export_validation.json"
)

MAX_LENGTH = 1024
BATCH_SIZE = 8
OPSET_VERSION = 17


class RerankerLogitsModel(nn.Module):
    """Expose only the sequence-classification logits for ONNX export."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw classification logits; sigmoid is applied separately."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


def load_baseline(path: Path) -> tuple[str, list[dict[str, Any]], dict[str, float]]:
    """Load the exact query, passages, and reference scores from the baseline."""
    if not path.is_file():
        raise FileNotFoundError(f"Baseline benchmark not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    candidates = data["candidates"]
    ranking = data["ranking"]

    reference_scores = {
        item["chunk_id"]: float(item["reranker_score"])
        for item in ranking
    }

    if not candidates or not reference_scores:
        raise ValueError("Baseline contains no candidates or ranking scores.")

    missing = [
        item["chunk_id"]
        for item in candidates
        if item["chunk_id"] not in reference_scores
    ]
    if missing:
        raise ValueError(
            f"Baseline scores are missing for candidate IDs: {missing}"
        )

    return data["query"], candidates, reference_scores


def tokenize_batch(
    tokenizer: Any,
    query: str,
    passages: list[str],
) -> dict[str, torch.Tensor]:
    """Tokenize query–passage pairs using the reranker's original settings."""
    encoded = tokenizer(
        [query] * len(passages),
        passages,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # XLM-RoBERTa uses input_ids and attention_mask for this classifier.
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def collect_pytorch_scores(
    model: nn.Module,
    tokenizer: Any,
    query: str,
    candidates: list[dict[str, Any]],
) -> np.ndarray:
    """Run the original PyTorch model in the baseline's batch size."""
    scores: list[np.ndarray] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[start : start + BATCH_SIZE]
            passages = [item["text"] for item in batch]
            encoded = tokenize_batch(tokenizer, query, passages)

            logits = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            ).logits

            scores.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())

    return np.concatenate(scores).astype(np.float32)


def collect_onnx_scores(
    session: ort.InferenceSession,
    tokenizer: Any,
    query: str,
    candidates: list[dict[str, Any]],
) -> np.ndarray:
    """Run the exported model using ONNX Runtime's CPU execution provider."""
    scores: list[np.ndarray] = []

    for start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[start : start + BATCH_SIZE]
        passages = [item["text"] for item in batch]
        encoded = tokenize_batch(tokenizer, query, passages)

        outputs = session.run(
            ["logits"],
            {
                "input_ids": encoded["input_ids"].numpy(),
                "attention_mask": encoded["attention_mask"].numpy(),
            },
        )

        # CrossEncoder's inspected activation is sigmoid, so reproduce it
        # outside ONNX to compare scores on the same scale.
        logits = outputs[0]
        scores.append(
            (1.0 / (1.0 + np.exp(-logits))).squeeze(-1).astype(np.float32)
        )

    return np.concatenate(scores)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and validate the BGE reranker as FP32 ONNX."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing ONNX model and validation report.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    torch.set_num_threads(8)
    torch.set_num_interop_threads(4)

    LOGGER.info(
        "CPU threads | OMP=%s | OpenBLAS=%s | PyTorch intra=%d | inter=%d",
        os.environ["OMP_NUM_THREADS"],
        os.environ["OPENBLAS_NUM_THREADS"],
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )

    if not args.overwrite and (ONNX_PATH.exists() or REPORT_PATH.exists()):
        raise FileExistsError(
            "An ONNX model or validation report already exists. "
            "Inspect it first, or rerun with --overwrite."
        )

    query, candidates, reference_scores = load_baseline(BASELINE_PATH)
    LOGGER.info("Loaded %d baseline candidates.", len(candidates))

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading original CrossEncoder on CPU...")
    cross_encoder = CrossEncoder(
        MODEL_NAME,
        device="cpu",
        max_length=MAX_LENGTH,
    )

    hf_model = cross_encoder.model
    tokenizer = cross_encoder.tokenizer
    hf_model.eval()

    if hf_model.config.model_type != "xlm-roberta":
        raise RuntimeError(
            f"Unexpected model type: {hf_model.config.model_type}"
        )
    if hf_model.config.num_labels != 1:
        raise RuntimeError(
            f"Expected one output logit, got {hf_model.config.num_labels}."
        )

    LOGGER.info("Preparing a representative export input...")
    sample_passages = [
        item["text"] for item in candidates[: min(2, len(candidates))]
    ]
    sample = tokenize_batch(tokenizer, query, sample_passages)

    export_model = RerankerLogitsModel(hf_model).eval()

    LOGGER.info("Exporting FP32 ONNX model to %s", ONNX_PATH)

    # Dynamic axes allow inference with varying batch sizes and token lengths.
    # External data is enabled because this model may exceed ONNX's
    # single-protobuf-file size limit.
    with torch.inference_mode():
        torch.onnx.export(
            export_model,
            (sample["input_ids"], sample["attention_mask"]),
            str(ONNX_PATH),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            export_params=True,
            external_data=True,
            dynamo=False,
        )

    LOGGER.info("Checking ONNX model structure...")
    onnx.checker.check_model(str(ONNX_PATH))

    LOGGER.info("Creating CPU-only ONNX Runtime session...")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 8
    session_options.inter_op_num_threads = 4
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(ONNX_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    if session.get_providers()[0] != "CPUExecutionProvider":
        raise RuntimeError(
            f"Unexpected ONNX provider selection: {session.get_providers()}"
        )

    LOGGER.info("Comparing PyTorch and ONNX scores on identical inputs...")
    pytorch_scores = collect_pytorch_scores(
        hf_model, tokenizer, query, candidates
    )
    onnx_scores = collect_onnx_scores(
        session, tokenizer, query, candidates
    )

    baseline_scores = np.asarray(
        [
            reference_scores[item["chunk_id"]]
            for item in candidates
        ],
        dtype=np.float32,
    )

    pytorch_vs_onnx = np.abs(pytorch_scores - onnx_scores)
    baseline_vs_pytorch = np.abs(baseline_scores - pytorch_scores)

    report = {
        "model": MODEL_NAME,
        "architecture": hf_model.config.architectures,
        "model_type": hf_model.config.model_type,
        "precision": "FP32",
        "activation": "sigmoid",
        "onnx_path": str(ONNX_PATH.relative_to(PROJECT_ROOT)),
        "onnx_opset": OPSET_VERSION,
        "execution_provider": session.get_providers()[0],
        "candidate_count": len(candidates),
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "pytorch_vs_onnx": {
            "max_absolute_score_error": float(pytorch_vs_onnx.max()),
            "mean_absolute_score_error": float(pytorch_vs_onnx.mean()),
        },
        "baseline_vs_pytorch": {
            "max_absolute_score_error": float(baseline_vs_pytorch.max()),
            "mean_absolute_score_error": float(baseline_vs_pytorch.mean()),
        },
        "scores": [
            {
                "chunk_id": item["chunk_id"],
                "baseline_score": float(baseline_scores[index]),
                "pytorch_score": float(pytorch_scores[index]),
                "onnx_score": float(onnx_scores[index]),
                "absolute_error": float(pytorch_vs_onnx[index]),
            }
            for index, item in enumerate(candidates)
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("PyTorch vs ONNX max score error: %.8g", pytorch_vs_onnx.max())
    LOGGER.info("PyTorch vs ONNX mean score error: %.8g", pytorch_vs_onnx.mean())
    LOGGER.info(
        "Baseline vs PyTorch max score error: %.8g",
        baseline_vs_pytorch.max(),
    )
    LOGGER.info("Validation report saved to %s", REPORT_PATH)
    LOGGER.info("FP32 ONNX export and validation completed.")


if __name__ == "__main__":
    main()
