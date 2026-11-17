
"""
Evaluate context-selection policies using saved retrieval diagnostics.

The experiment evaluates two production decisions:
    1. How many reranked candidates are passed to ContextBuilder.
    2. How many chunks from one paper ContextBuilder may retain.

It reuses saved passage text and metadata across the complete
answerable benchmark, without loading retrieval or reranking models.

The production ContextBuilder implementation is used to preserve
candidate validation, duplicate removal, per-paper selection, and
token-budget behavior.

The experiment uses the Qwen3-14B tokenizer and the production
6,000-token evidence-context budget. It does not load the generator.
"""

from __future__ import annotations
from scilitrag.core.paths import EVALUATION_DIR

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from scilitrag.pipeline.context_builder import ContextBuilder
from transformers import AutoTokenizer


LOGGER = logging.getLogger(__name__)



DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/evaluation/retrieval_results_full_diagnostics.json"
)

OUTPUT_DIR = EVALUATION_DIR

# Each tuple represents (reranking depth, maximum chunks per paper).
EXPERIMENTS = (
    (5, 3),
    (10, 3),
    (10, 5),
    (10, 10),
)

# Retain passages whose reranker score is within this margin
# of the highest score for the current query.
ADAPTIVE_SCORE_DELTA = 0.20

# Match the production evidence-context budget.
EXPERIMENT_TOKEN_BUDGET = 6000

# Use the same tokenizer as the production answer generator.
TOKENIZER_NAME = "Qwen/Qwen3-14B"


def load_report(path: Path) -> dict[str, Any]:
    """Load and validate the saved retrieval evaluation report."""

    if not path.is_file():
        raise FileNotFoundError(f"Evaluation report not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report.get("ranking_diagnostics"), dict):
        raise ValueError(
            "The evaluation report has no valid ranking_diagnostics."
        )

    return report



def evaluate_question(
    question_id: str,
    question_data: dict[str, Any],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Evaluate fixed and adaptive selection policies for one question."""

    gold_chunk_ids = set(question_data["gold_chunk_ids"])

    # Use reranked candidates, preserving their descending score order.
    ranked_candidates = question_data["systems"]["reranked"][
        "top_10_passages"
    ]

    results: list[dict[str, Any]] = []

    def evaluate_policy(
        policy_name: str,
        candidates: list[dict[str, Any]],
        paper_cap: int,
        candidate_depth: int,
        score_delta: float | None = None,
    ) -> dict[str, Any]:
        """Apply ContextBuilder and measure the resulting evidence coverage."""

        builder = ContextBuilder(
            tokenizer=tokenizer,
            max_tokens=EXPERIMENT_TOKEN_BUDGET,
            max_chunks_per_paper=paper_cap,
        )

        assembled = builder.build(candidates)

        selected_chunk_ids = {
            evidence.chunk_id for evidence in assembled.evidence
        }

        retrieved_gold = gold_chunk_ids.intersection(
            candidate["chunk_id"] for candidate in candidates
        )

        retained_gold = gold_chunk_ids.intersection(
            selected_chunk_ids
        )

        # This includes all retrieved gold evidence not present in the
        # final context, regardless of whether the cap or token budget
        # caused its exclusion.
        not_retained_gold = retrieved_gold - retained_gold

        return {
            "question_id": question_id,
            "policy": policy_name,
            "candidate_depth": candidate_depth,
            "max_chunks_per_paper": paper_cap,
            "score_delta": score_delta,
            "input_candidates": len(candidates),
            "selected_passages": len(assembled.evidence),
            "retrieved_gold_count": len(retrieved_gold),
            "retained_gold_count": len(retained_gold),
            "gold_total": len(gold_chunk_ids),
            "gold_chunks_retained": sorted(retained_gold),
            "gold_chunks_not_retained": sorted(not_retained_gold),
            "gold_coverage": (
                len(retained_gold) / len(gold_chunk_ids)
                if gold_chunk_ids
                else None
            ),
            "token_count": assembled.token_count,
        }

    # Evaluate the existing fixed-depth and per-paper-cap configurations.
    for candidate_depth, paper_cap in EXPERIMENTS:
        candidates = ranked_candidates[:candidate_depth]

        results.append(
            evaluate_policy(
                policy_name=f"fixed_depth_{candidate_depth}_cap_{paper_cap}",
                candidates=candidates,
                paper_cap=paper_cap,
                candidate_depth=candidate_depth,
            )
        )

    # Adaptive policy: retain candidates while their scores remain
    # within the configured margin of the highest score for this query.
    adaptive_candidates = ranked_candidates[:10]

    if adaptive_candidates:
        best_score = float(
            adaptive_candidates[0]["reranker_score"]
        )

        selected_candidates = []

        for candidate in adaptive_candidates:
            score = float(candidate["reranker_score"])

            # Stop at the first candidate outside the relative score margin.
            if best_score - score > ADAPTIVE_SCORE_DELTA:
                break

            selected_candidates.append(candidate)

        results.append(
            evaluate_policy(
                policy_name="adaptive_relative_score",
                candidates=selected_candidates,
                paper_cap=10,
                candidate_depth=10,
                score_delta=ADAPTIVE_SCORE_DELTA,
            )
        )

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a compact comparison grouped by question and configuration."""

    print("\n" + "=" * 88)
    print("CONTEXT SELECTION EXPERIMENT")
    print("=" * 88)

    print(
        f"{'Question':<10}"
        f"{'Policy':<30}"
        f"{'Selected':>10}"
        f"{'Gold kept':>12}"
        f"{'Coverage':>11}"
        f"{'Tokens':>9}"
    )

    print("-" * 92)

    for result in results:
        coverage = result["gold_coverage"]

        coverage_text = (
            f"{coverage:.1%}" if coverage is not None else "N/A"
        )

        print(
            f"{result['question_id']:<10}"
            f"{result['policy']:<30}"
            f"{result['selected_passages']:>10}"
            f"{result['retained_gold_count']:>8}/"
            f"{result['gold_total']:<4}"
            f"{coverage_text:>11}"
            f"{result['token_count']:>9}"
        )

    print("=" * 92)

    print("\nRetrieved gold evidence not retained in context:")

    for result in results:
        not_retained = result["gold_chunks_not_retained"]

        if not_retained:
            print(
                f"  {result['question_id']} "
                f"({result['policy']}): {not_retained}"
            )



def main() -> None:
    """Run the saved-ranking context-selection experiment."""

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    report = load_report(DEFAULT_INPUT)

    # Load only the tokenizer. QwenGenerator is deliberately not
    # instantiated because that would also load the 14B model.
    LOGGER.info("Loading tokenizer: %s", TOKENIZER_NAME)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    all_results: list[dict[str, Any]] = []

    try:
        for question_id, question_data in report[
            "ranking_diagnostics"
        ].items():
            LOGGER.info("Evaluating %s", question_id)

            all_results.extend(
                evaluate_question(
                    question_id,
                    question_data,
                    tokenizer,
                )
            )

    finally:
        # Release the tokenizer reference after all experiments finish.
        del tokenizer

    print_summary(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = (
        OUTPUT_DIR
        / f"context_selection_results_{timestamp}.json"
    )

    output = {
        "metadata": {
            "source_report": DEFAULT_INPUT.name,
            "experiment": "fixed_vs_adaptive_context_selection",
            "questions_evaluated": len(
                report["ranking_diagnostics"]
            ),
            "experiments": [
                {
                    "policy": f"fixed_depth_{depth}_cap_{cap}",
                    "candidate_depth": depth,
                    "max_chunks_per_paper": cap,
                }
                for depth, cap in EXPERIMENTS
            ] + [
                {
                    "policy": "adaptive_relative_score",
                    "candidate_depth": 10,
                    "max_chunks_per_paper": 10,
                    "score_delta": ADAPTIVE_SCORE_DELTA,
                }
            ],
            "token_budget": EXPERIMENT_TOKEN_BUDGET,
            "tokenizer": TOKENIZER_NAME,
        },
        "results": all_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nSaved experiment report: {output_path}")


if __name__ == "__main__":
    main()
