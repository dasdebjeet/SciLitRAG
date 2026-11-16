
"""
Evaluate generated answer quality under production and adaptive context policies.

The experiment:
    - Reuses saved retrieval and reranking diagnostics.
    - Compares production top-5/cap-3 against adaptive top-10 selection.
    - Uses the production ContextBuilder, prompt, citation resolver, and Qwen generator.
    - Loads Qwen3-14B once and reuses it for all generations.
    - Records answers, evidence, citations, token counts, and generation latency.

No retrieval, embedding, or reranking models are loaded.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from answer_pipeline import (
    SYSTEM_PROMPT,
    build_user_prompt,
    resolve_citations,
)
from context_builder import ContextBuilder
from generator import QwenGenerator


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_REPORT = (
    PROJECT_ROOT
    / "data/evaluation/retrieval_results_full_diagnostics.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data/evaluation"

QUESTION_IDS = ("q003", "q004", "q008", "q013")

CONTEXT_TOKEN_BUDGET = 6000
MAX_NEW_TOKENS = 1200
ADAPTIVE_SCORE_DELTA = 0.20

# The two policies being compared.
POLICIES = (
    {
        "name": "production",
        "candidate_depth": 5,
        "paper_cap": 3,
    },
    {
        "name": "adaptive",
        "candidate_depth": 10,
        "paper_cap": 10,
    },
)


def load_diagnostics() -> dict[str, Any]:
    """Load the previously generated retrieval diagnostics."""

    if not INPUT_REPORT.is_file():
        raise FileNotFoundError(
            f"Retrieval diagnostics not found: {INPUT_REPORT}"
        )

    with INPUT_REPORT.open("r", encoding="utf-8") as file:
        report = json.load(file)

    diagnostics = report.get("ranking_diagnostics")

    if not isinstance(diagnostics, dict):
        raise ValueError(
            "Input report does not contain ranking_diagnostics."
        )

    missing = [
        question_id
        for question_id in QUESTION_IDS
        if question_id not in diagnostics
    ]

    if missing:
        raise ValueError(
            f"Questions missing from retrieval diagnostics: {missing}"
        )

    return diagnostics


def select_adaptive_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Apply the same relative reranker-score rule used in the
    context-selection experiment.

    Selection stops at the first candidate whose score is more
    than ADAPTIVE_SCORE_DELTA below the highest score.
    """

    if not candidates:
        return []

    best_score = float(candidates[0]["reranker_score"])
    selected = []

    for candidate in candidates:
        score = float(candidate["reranker_score"])

        if best_score - score > ADAPTIVE_SCORE_DELTA:
            break

        selected.append(candidate)

    return selected


def build_policy_context(
    policy: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
    tokenizer: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    """Select candidates and assemble evidence using production logic."""

    if policy["name"] == "adaptive":
        candidates = select_adaptive_candidates(
            ranked_candidates[:policy["candidate_depth"]]
        )
    else:
        candidates = ranked_candidates[:policy["candidate_depth"]]

    builder = ContextBuilder(
        tokenizer=tokenizer,
        max_tokens=CONTEXT_TOKEN_BUDGET,
        max_chunks_per_paper=policy["paper_cap"],
    )

    assembled_context = builder.build(candidates)

    return assembled_context, candidates


def count_prompt_tokens(
    tokenizer: Any,
    prompt: str,
) -> int:
    """
    Count tokens using the same Qwen chat template as QwenGenerator.

    This includes the system prompt, user prompt, and chat-template
    tokens, rather than counting only the evidence passages.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )

    return int(model_inputs["input_ids"].shape[1])


def serialize_evidence(evidence: list[Any]) -> list[dict[str, Any]]:
    """Convert Evidence dataclasses into JSON-serializable records."""

    return [
        {
            "evidence_id": item.evidence_id,
            "chunk_id": item.chunk_id,
            "arxiv_id": item.arxiv_id,
            "title": item.title,
            "page_number": item.page_number,
            "reranker_score": item.reranker_score,
            "text": item.text,
        }
        for item in evidence
    ]


def evaluate_one(
    question_id: str,
    question_data: dict[str, Any],
    policy: dict[str, Any],
    generator: QwenGenerator,
) -> dict[str, Any]:
    """Generate and record one answer under one context-selection policy."""

    question = question_data.get("question")

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            f"{question_id} has no valid 'question' field in diagnostics."
        )

    gold_chunk_ids = set(question_data["gold_chunk_ids"])

    ranked_candidates = question_data["systems"]["reranked"][
        "top_10_passages"
    ]

    assembled_context, selected_candidates = build_policy_context(
        policy=policy,
        ranked_candidates=ranked_candidates,
        tokenizer=generator.tokenizer,
    )

    selected_chunk_ids = {
        item.chunk_id for item in assembled_context.evidence
    }

    retained_gold = gold_chunk_ids.intersection(selected_chunk_ids)

    result: dict[str, Any] = {
        "question_id": question_id,
        "question": question,
        "policy": policy["name"],
        "candidate_depth": policy["candidate_depth"],
        "max_chunks_per_paper": policy["paper_cap"],
        "adaptive_score_delta": (
            ADAPTIVE_SCORE_DELTA
            if policy["name"] == "adaptive"
            else None
        ),
        "input_candidates": len(selected_candidates),
        "selected_passages": len(assembled_context.evidence),
        "gold_total": len(gold_chunk_ids),
        "retained_gold_count": len(retained_gold),
        "gold_coverage": (
            len(retained_gold) / len(gold_chunk_ids)
            if gold_chunk_ids
            else None
        ),
        "gold_chunks_retained": sorted(retained_gold),
        "gold_chunks_not_retained": sorted(
            gold_chunk_ids - retained_gold
        ),
        "context_tokens": assembled_context.token_count,
        "evidence": serialize_evidence(assembled_context.evidence),
    }

    if not assembled_context.evidence:
        result["error"] = "No usable evidence was assembled."
        return result

    prompt = build_user_prompt(
        question=question,
        assembled_context=assembled_context,
    )

    result["full_prompt_tokens"] = count_prompt_tokens(
        generator.tokenizer,
        prompt,
    )

    LOGGER.info(
        "Generating %s | %s | evidence=%d | prompt_tokens=%d",
        question_id,
        policy["name"],
        len(assembled_context.evidence),
        result["full_prompt_tokens"],
    )

    # Measure actual generation time, excluding context construction.
    torch.cuda.synchronize(0)
    start_time = time.perf_counter()

    generated_answer = generator.generate(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    torch.cuda.synchronize(0)
    generation_seconds = time.perf_counter() - start_time

    final_answer, resolved_citations = resolve_citations(
        answer=generated_answer,
        evidence=assembled_context.evidence,
    )

    result.update(
        {
            "answer": final_answer,
            "generation_seconds": round(generation_seconds, 3),
            "generated_answer_tokens": len(
                generator.tokenizer.encode(
                    generated_answer,
                    add_special_tokens=False,
                )
            ),
            "citations": [
                {
                    "evidence_id": item.evidence_id,
                    "chunk_id": item.chunk_id,
                    "arxiv_id": item.arxiv_id,
                    "title": item.title,
                    "page_number": item.page_number,
                }
                for item in resolved_citations
            ],
            "citation_count": len(resolved_citations),
        }
    )

    return result


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a compact summary of the generated-answer experiment."""

    print("\n" + "=" * 100)
    print("ANSWER QUALITY EXPERIMENT")
    print("=" * 100)

    print(
        f"{'Question':<10}"
        f"{'Policy':<14}"
        f"{'Evidence':>10}"
        f"{'Gold kept':>12}"
        f"{'Context':>10}"
        f"{'Latency':>11}"
        f"{'Citations':>11}"
    )

    print("-" * 100)

    for result in results:
        coverage = result.get("gold_coverage")
        coverage_text = (
            f"{coverage:.1%}" if coverage is not None else "N/A"
        )

        latency = result.get("generation_seconds", 0.0)

        print(
            f"{result['question_id']:<10}"
            f"{result['policy']:<14}"
            f"{result['selected_passages']:>10}"
            f"{result['retained_gold_count']:>8}/"
            f"{result['gold_total']:<4}"
            f"{result['context_tokens']:>10}"
            f"{latency:>10.1f}s"
            f"{result.get('citation_count', 0):>11}"
        )

    print("=" * 100)

    failed = [item for item in results if "error" in item]

    if failed:
        print(f"\nFailed evaluations: {len(failed)}")
        for item in failed:
            print(f"  {item['question_id']} / {item['policy']}: {item['error']}")


def main() -> None:
    """Run the eight-generation answer-quality experiment."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    diagnostics = load_diagnostics()

    generator = None
    results: list[dict[str, Any]] = []

    try:
        # Load the largest model only once and reuse it for all eight answers.
        LOGGER.info("Loading Qwen3-14B for answer-quality evaluation.")

        generator = QwenGenerator(
            max_new_tokens=MAX_NEW_TOKENS,
        )

        for question_id in QUESTION_IDS:
            question_data = diagnostics[question_id]

            for policy in POLICIES:
                try:
                    result = evaluate_one(
                        question_id=question_id,
                        question_data=question_data,
                        policy=policy,
                        generator=generator,
                    )

                except Exception as exc:
                    LOGGER.exception(
                        "Evaluation failed: %s / %s",
                        question_id,
                        policy["name"],
                    )

                    result = {
                        "question_id": question_id,
                        "policy": policy["name"],
                        "error": str(exc),
                    }

                results.append(result)

    finally:
        # Release Qwen and its GPU allocations even if an evaluation fails.
        if generator is not None:
            generator.close()

    print_summary(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        OUTPUT_DIR / f"answer_quality_results_{timestamp}.json"
    )

    output = {
        "metadata": {
            "experiment": "production_vs_adaptive_answer_quality",
            "questions": list(QUESTION_IDS),
            "policies": list(POLICIES),
            "adaptive_score_delta": ADAPTIVE_SCORE_DELTA,
            "context_token_budget": CONTEXT_TOKEN_BUDGET,
            "max_new_tokens": MAX_NEW_TOKENS,
            "model": "Qwen/Qwen3-14B",
            "decoding": "greedy; thinking disabled",
            "system_prompt": "Production SYSTEM_PROMPT",
            "source_report": INPUT_REPORT.name,
            "completed_at": datetime.now().isoformat(),
        },
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nSaved answer-quality report: {output_path}")


if __name__ == "__main__":
    main()
