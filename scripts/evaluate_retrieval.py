
"""
Evaluate SciLitRAG retrieval against a manually verified benchmark.

Evaluation stages:
    1. Hybrid retrieval: BM25 + dense retrieval + Reciprocal Rank Fusion.
    2. Cross-encoder reranking: reorder the hybrid candidate set.

Metrics:
    - Recall@5 and Recall@10
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (nDCG@5)

Evaluation levels:
    - Paper-level: a retrieved paper is relevant when its arxiv_id
      appears in the question's expected_paper_ids.
    - Chunk-level: a retrieved passage is relevant when its chunk_id
      appears in the question's expected_chunk_ids.

Unanswerable questions:
    These are excluded from standard retrieval metrics because they
    intentionally have no gold paper or chunk IDs. Their retrieved
    passages are recorded separately for manual inspection.

Resource management:
    Hybrid retrieval and reranking are run sequentially to avoid keeping
    the embedding model and cross-encoder resident on the GPU together.

Run:
    python scripts/evaluate_retrieval.py
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from hybrid_retriever import HybridRetriever
from reranker import CrossEncoderReranker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BENCHMARK = (
    PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"

DEFAULT_RETRIEVAL_K = 20
DEFAULT_RERANKER_TOP_K = 10

EVALUATION_KS = (5, 10)
NDCG_K = 5

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

LOGGER = logging.getLogger("evaluate_retrieval")

REQUIRED_FIELDS = {
    "question_id",
    "question",
    "question_type",
    "answerable",
    "expected_paper_ids",
    "expected_chunk_ids",
    "verified",
}


# ---------------------------------------------------------------------------
# Benchmark loading and validation
# ---------------------------------------------------------------------------

def load_benchmark(path: Path) -> list[dict[str, Any]]:
    """Load and validate the JSONL evaluation benchmark.

    Every question must have a unique ID and be manually verified.
    Answerable questions must contain gold paper and chunk identifiers.
    Unanswerable questions must not contain gold retrieval identifiers.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Benchmark file does not exist: {path}"
        )

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                raise ValueError(
                    f"Blank benchmark line at line {line_number}."
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            missing = REQUIRED_FIELDS - record.keys()

            if missing:
                raise ValueError(
                    f"Line {line_number} is missing fields: "
                    f"{sorted(missing)}"
                )

            question_id = record["question_id"]

            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError(
                    f"Invalid question_id at line {line_number}."
                )

            if question_id in seen_ids:
                raise ValueError(
                    f"Duplicate question ID: {question_id}"
                )

            seen_ids.add(question_id)

            if record["verified"] is not True:
                raise ValueError(
                    f"{question_id} is not manually verified."
                )

            if not isinstance(record["question"], str) or not record["question"].strip():
                raise ValueError(
                    f"{question_id} has an empty question."
                )

            if not isinstance(record["answerable"], bool):
                raise ValueError(
                    f"{question_id} has an invalid answerable value."
                )

            paper_ids = record["expected_paper_ids"]
            chunk_ids = record["expected_chunk_ids"]

            if not isinstance(paper_ids, list) or not isinstance(chunk_ids, list):
                raise ValueError(
                    f"{question_id}: expected paper/chunk IDs must be lists."
                )

            if record["answerable"]:
                if not paper_ids or not chunk_ids:
                    raise ValueError(
                        f"{question_id}: answerable question requires "
                        "gold paper and chunk IDs."
                    )
            elif paper_ids or chunk_ids:
                raise ValueError(
                    f"{question_id}: unanswerable question must not "
                    "contain gold retrieval IDs."
                )

            records.append(record)

    if not records:
        raise ValueError("Benchmark contains no questions.")

    LOGGER.info(
        "Loaded %d verified questions | answerable=%d | unanswerable=%d",
        len(records),
        sum(record["answerable"] for record in records),
        sum(not record["answerable"] for record in records),
    )

    return records


# ---------------------------------------------------------------------------
# Ranking preparation
# ---------------------------------------------------------------------------

def deduplicate_ranking(
    results: list[dict[str, Any]],
    identifier_field: str,
) -> list[dict[str, Any]]:
    """Remove duplicate papers or chunks while preserving ranking order.

    Paper-level evaluation counts each paper only once, even when multiple
    passages from the same paper occur in the retrieved result list.
    """

    unique_results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for result in results:
        identifier = result.get(identifier_field)

        if not isinstance(identifier, str) or not identifier:
            continue

        if identifier in seen_ids:
            continue

        seen_ids.add(identifier)
        unique_results.append(result)

    return unique_results


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def calculate_metrics(
    results: list[dict[str, Any]],
    gold_ids: set[str],
    identifier_field: str,
) -> dict[str, float]:
    """Calculate binary-relevance retrieval metrics for one question.

    Recall@K:
        Relevant retrieved items in the first K positions divided by
        the total number of gold relevant items.

    MRR@10:
        Reciprocal rank of the first relevant item within the top 10
        retrieved positions. Uses the same cutoff for both systems.

    nDCG@5:
        Discounted cumulative gain at five positions divided by the
        ideal DCG for the available gold relevant items.
    """

    if not gold_ids:
        raise ValueError("Gold identifier set cannot be empty.")

    ranking = deduplicate_ranking(results, identifier_field)

    relevance = [
        1 if result[identifier_field] in gold_ids else 0
        for result in ranking
    ]

    metrics: dict[str, float] = {}

    # Recall measures how much of the known relevant evidence was retrieved.
    for k in EVALUATION_KS:
        retrieved_relevant = sum(relevance[:k])

        metrics[f"recall@{k}"] = (
            retrieved_relevant / len(gold_ids)
        )

    # MRR@10 evaluates both retrieval systems at the same ranking depth.
    # This prevents the hybrid system from receiving credit for relevant
    # results at ranks 11-20 when reranked results contain only 10 items.
    mrr_cutoff = 10

    first_relevant_rank = next(
        (
            rank
            for rank, relevant in enumerate(
                relevance[:mrr_cutoff],
                start=1,
            )
            if relevant
        ),
        None,
    )

    metrics[f"mrr@{mrr_cutoff}"] = (
        1.0 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )

    # Calculate nDCG using 1-based ranks:
    # rank 1 has discount log2(2), rank 2 has log2(3), etc.
    dcg = sum(
        relevant / math.log2(rank + 1)
        for rank, relevant in enumerate(
            relevance[:NDCG_K],
            start=1,
        )
    )

    ideal_relevance = [1] * min(len(gold_ids), NDCG_K)

    idcg = sum(
        relevant / math.log2(rank + 1)
        for rank, relevant in enumerate(
            ideal_relevance,
            start=1,
        )
    )

    metrics[f"ndcg@{NDCG_K}"] = (
        dcg / idcg if idcg > 0 else 0.0
    )

    return metrics


def evaluate_question(
    results: list[dict[str, Any]],
    question: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Evaluate one question at both paper and chunk levels."""

    gold_papers = set(question["expected_paper_ids"])
    gold_chunks = set(question["expected_chunk_ids"])

    return {
        "paper_level": calculate_metrics(
            results,
            gold_papers,
            "arxiv_id",
        ),
        "chunk_level": calculate_metrics(
            results,
            gold_chunks,
            "chunk_id",
        ),
    }


def aggregate_metrics(
    per_question_results: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Calculate macro-averaged metrics across answerable questions.

    Each question contributes equally, regardless of how many gold papers
    or chunks it contains.
    """

    if not per_question_results:
        return {}

    aggregated: dict[str, dict[str, float]] = {}

    for level in ("paper_level", "chunk_level"):

        metric_names = per_question_results[
            next(iter(per_question_results))
        ][level].keys()

        aggregated[level] = {}

        for metric_name in metric_names:
            values = [
                question_result[level][metric_name]
                for question_result in per_question_results.values()
            ]

            aggregated[level][metric_name] = (
                sum(values) / len(values)
            )

    return aggregated


# ---------------------------------------------------------------------------
# Evaluation execution
# ---------------------------------------------------------------------------

def run_hybrid_retrieval(
    questions: list[dict[str, Any]],
    retrieval_k: int,
    device: str,
) -> dict[str, list[dict[str, Any]]]:
    """Retrieve and retain hybrid candidates for every benchmark question.

    The hybrid retriever is closed before the reranker is initialized,
    allowing the GPU embedding model to be released.
    """

    LOGGER.info("Initializing hybrid retriever.")

    retriever = HybridRetriever(device=device)
    retrieved: dict[str, list[dict[str, Any]]] = {}

    try:
        for index, question in enumerate(questions, start=1):
            question_id = question["question_id"]

            LOGGER.info(
                "[%d/%d] Hybrid retrieval | %s",
                index,
                len(questions),
                question_id,
            )

            results = retriever.search(
                query=question["question"],
                top_k=retrieval_k,
                retrieval_k=retrieval_k,
            )

            retrieved[question_id] = results

            LOGGER.info(
                "%s | retrieved=%d | gold_papers=%d | gold_chunks=%d",
                question_id,
                len(results),
                len(question["expected_paper_ids"]),
                len(question["expected_chunk_ids"]),
            )

    finally:
        retriever.close()
        del retriever
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize(0)
            torch.cuda.empty_cache()

        LOGGER.info("Hybrid retrieval resources released.")

    return retrieved


def run_reranking(
    questions: list[dict[str, Any]],
    hybrid_results: dict[str, list[dict[str, Any]]],
    reranker_top_k: int,
    device: str,
) -> dict[str, list[dict[str, Any]]]:
    """Rerank the saved hybrid candidates using the cross-encoder."""

    LOGGER.info("Initializing cross-encoder reranker.")

    reranker = CrossEncoderReranker(device=device)
    reranked: dict[str, list[dict[str, Any]]] = {}

    try:
        for index, question in enumerate(questions, start=1):
            question_id = question["question_id"]
            candidates = hybrid_results[question_id]

            LOGGER.info(
                "[%d/%d] Reranking | %s | candidates=%d",
                index,
                len(questions),
                question_id,
                len(candidates),
            )

            if not candidates:
                reranked[question_id] = []
                continue

            results = reranker.rerank(
                query=question["question"],
                candidates=candidates,
                top_k=min(reranker_top_k, len(candidates)),
            )

            reranked[question_id] = results

    finally:
        reranker.close()
        del reranker
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize(0)
            torch.cuda.empty_cache()

        LOGGER.info("Cross-encoder resources released.")

    return reranked


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_evaluation_report(
    questions: list[dict[str, Any]],
    hybrid_results: dict[str, list[dict[str, Any]]],
    reranked_results: dict[str, list[dict[str, Any]]],
    retrieval_k: int,
    reranker_top_k: int,
) -> dict[str, Any]:
    """Build summary metrics and question-level evaluation records."""

    answerable_questions = [
        question for question in questions
        if question["answerable"]
    ]

    unanswerable_questions = [
        question for question in questions
        if not question["answerable"]
    ]

    systems = {
        "hybrid": hybrid_results,
        "reranked": reranked_results,
    }

    per_question: dict[str, Any] = {}
    summaries: dict[str, Any] = {}

    for system_name, system_results in systems.items():

        question_metrics: dict[
            str, dict[str, dict[str, float]]
        ] = {}

        for question in answerable_questions:
            question_id = question["question_id"]

            question_metrics[question_id] = evaluate_question(
                system_results[question_id],
                question,
            )

        summaries[system_name] = {
            "answerable_question_count": len(answerable_questions),
            "metrics": aggregate_metrics(question_metrics),
        }

        for question_id, metrics in question_metrics.items():
            per_question.setdefault(question_id, {})[
                system_name
            ] = metrics

    # Preserve diagnostics for questions without known relevant passages.
    unanswerable_diagnostics = []

    for question in unanswerable_questions:
        question_id = question["question_id"]

        hybrid = hybrid_results[question_id]
        reranked = reranked_results[question_id]

        unanswerable_diagnostics.append({
            "question_id": question_id,
            "question": question["question"],
            "hybrid": {
                "retrieved_count": len(hybrid),
                "unique_papers": sorted({
                    item["arxiv_id"]
                    for item in hybrid
                    if item.get("arxiv_id")
                }),
                "top_5_chunk_ids": [
                    item.get("chunk_id")
                    for item in hybrid[:5]
                ],
            },
            "reranked": {
                "retrieved_count": len(reranked),
                "unique_papers": sorted({
                    item["arxiv_id"]
                    for item in reranked
                    if item.get("arxiv_id")
                }),
                "top_5_chunk_ids": [
                    item.get("chunk_id")
                    for item in reranked[:5]
                ],
            },
        })

    
    # Preserve ranked passages for selected questions so retrieval
    # regressions can be investigated without changing evaluation metrics.
    
    # Save detailed rankings for every answerable question.
    # This allows downstream context-selection experiments to use
    # the complete benchmark without rerunning retrieval or reranking.
    diagnostic_question_ids = {
        question["question_id"]
        for question in answerable_questions
    }
    ranking_diagnostics: dict[str, Any] = {}

    for question in questions:
        question_id = question["question_id"]

        if question_id not in diagnostic_question_ids:
            continue

        # Gold chunk IDs come from the manually verified benchmark.
        gold_chunk_ids = question.get("expected_chunk_ids", [])

        ranking_diagnostics[question_id] = {
            "question": question["question"],
            "expected_paper_ids": question.get("expected_paper_ids", []),
            "gold_chunk_ids": gold_chunk_ids,
            "systems": {},
        }

        for system_name, system_results in systems.items():
            candidates = system_results[question_id]

            # Record the 1-based rank of every gold chunk. A null rank
            # means that the chunk was not retrieved by this system.
            chunk_ranks = {
                item.get("chunk_id"): rank
                for rank, item in enumerate(candidates, start=1)
                if item.get("chunk_id")
            }

            gold_chunk_ranks = {
                chunk_id: chunk_ranks.get(chunk_id)
                for chunk_id in gold_chunk_ids
            }

            # Retain the top 10 passages with their metadata and scores.
            top_passages = []

            for rank, item in enumerate(candidates[:10], start=1):
                passage = {
                    "rank": rank,
                    "chunk_id": item.get("chunk_id"),
                    "arxiv_id": item.get("arxiv_id"),
                    "title": item.get("title"),
                    "page_number": item.get("page_number"),
                    "text": item.get("text"),
                }

                # Keep whichever ranking signals are available at this stage.
                for score_key in (
                    "rrf_score",
                    "rrf_rank",
                    "reranker_score",
                    "bm25_rank",
                    "dense_rank",
                    "bm25_score",
                    "dense_score",
                ):
                    if score_key in item:
                        passage[score_key] = item[score_key]

                top_passages.append(passage)

            ranking_diagnostics[question_id]["systems"][system_name] = {
                "gold_chunk_ranks": gold_chunk_ranks,
                "top_10_passages": top_passages,
            }


    return {
        "metadata": {
            "project": "SciLitRAG",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "benchmark_question_count": len(questions),
            "answerable_question_count": len(answerable_questions),
            "unanswerable_question_count": len(unanswerable_questions),
            "retrieval_candidate_depth": retrieval_k,
            "reranker_output_depth": reranker_top_k,
            "evaluation_k": list(EVALUATION_KS),
            "ndcg_k": NDCG_K,
            "averaging": "macro average across answerable questions",
        },
        "summary": summaries,
        "per_question": per_question,
        "unanswerable_diagnostics": unanswerable_diagnostics,
        "ranking_diagnostics": ranking_diagnostics,
    }


def print_summary(report: dict[str, Any]) -> None:
    """Print a compact, readable comparison of retrieval performance."""

    print("\n" + "=" * 72)
    print("SCILITRAG RETRIEVAL EVALUATION")
    print("=" * 72)

    metadata = report["metadata"]

    print(
        f"Questions: {metadata['benchmark_question_count']} | "
        f"Answerable: {metadata['answerable_question_count']} | "
        f"Unanswerable: {metadata['unanswerable_question_count']}"
    )

    print(
        f"Candidate depth: {metadata['retrieval_candidate_depth']} | "
        f"Reranker output depth: {metadata['reranker_output_depth']}"
    )

    metric_names = (
        "recall@5",
        "recall@10",
        "mrr@10",
        "ndcg@5",
    )

    for level in ("paper_level", "chunk_level"):

        print(f"\n{level.replace('_', ' ').upper()}")
        print("-" * 72)

        print(
            f"{'Metric':<16}"
            f"{'Hybrid':>16}"
            f"{'Reranked':>16}"
            f"{'Change':>16}"
        )

        hybrid_metrics = report["summary"]["hybrid"]["metrics"][level]
        reranked_metrics = report["summary"]["reranked"]["metrics"][level]

        for metric_name in metric_names:
            hybrid_value = hybrid_metrics[metric_name]
            reranked_value = reranked_metrics[metric_name]

            print(
                f"{metric_name:<16}"
                f"{hybrid_value:>16.4f}"
                f"{reranked_value:>16.4f}"
                f"{reranked_value - hybrid_value:>+16.4f}"
            )

    print("\nUnanswerable question diagnostics")
    print("-" * 72)

    for diagnostic in report["unanswerable_diagnostics"]:
        print(
            f"{diagnostic['question_id']} | "
            f"hybrid papers={len(diagnostic['hybrid']['unique_papers'])} | "
            f"reranked papers={len(diagnostic['reranked']['unique_papers'])}"
        )

    print("=" * 72)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse evaluation configuration from command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate SciLitRAG hybrid retrieval and reranking."
    )

    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="Path to the verified benchmark JSONL file.",
    )

    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=DEFAULT_RETRIEVAL_K,
        help="Number of hybrid candidates retrieved per question.",
    )

    parser.add_argument(
        "--reranker-top-k",
        type=int,
        default=DEFAULT_RERANKER_TOP_K,
        help="Number of reranked results retained per question.",
    )

    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Device used by the embedding and reranking models.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit JSON output path.",
    )

    args = parser.parse_args()

    if args.retrieval_k < max(EVALUATION_KS):
        parser.error(
            f"--retrieval-k must be at least {max(EVALUATION_KS)}."
        )

    if args.reranker_top_k < max(EVALUATION_KS):
        parser.error(
            f"--reranker-top-k must be at least {max(EVALUATION_KS)}."
        )

    if args.reranker_top_k > args.retrieval_k:
        parser.error(
            "--reranker-top-k cannot exceed --retrieval-k."
        )

    return args


def main() -> None:
    """Run the complete benchmark evaluation and save its results."""

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
    )

    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA device."
        )

    questions = load_benchmark(args.benchmark)

    # Stage 1: collect hybrid candidates and release the embedding model.
    hybrid_results = run_hybrid_retrieval(
        questions=questions,
        retrieval_k=args.retrieval_k,
        device=args.device,
    )

    # Stage 2: load the cross-encoder only after hybrid retrieval is closed.
    reranked_results = run_reranking(
        questions=questions,
        hybrid_results=hybrid_results,
        reranker_top_k=args.reranker_top_k,
        device=args.device,
    )

    report = build_evaluation_report(
        questions=questions,
        hybrid_results=hybrid_results,
        reranked_results=reranked_results,
        retrieval_k=args.retrieval_k,
        reranker_top_k=args.reranker_top_k,
    )

    # Use a timestamped default path so earlier evaluation runs are preserved.
    output_path = args.output or (
        DEFAULT_OUTPUT_DIR
        / (
            "retrieval_results_"
            + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            + ".json"
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print_summary(report)

    print(f"\nFull evaluation report saved to:\n{output_path}")


if __name__ == "__main__":
    main()
