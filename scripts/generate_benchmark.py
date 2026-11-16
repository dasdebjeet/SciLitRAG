"""
Generate draft evaluation questions for the SciLitRAG benchmark.

The script:
    1. Loads paper metadata and extracted chunks from the existing corpus.
    2. Reads the verified benchmark and all existing candidate files.
    3. Selects papers not yet represented in the benchmark by default.
    4. Selects relevant source passages for each requested category.
    5. Uses the local QwenGenerator to propose benchmark questions.
    6. Maps model-provided source labels to real paper and chunk IDs.
    7. Rejects duplicate questions and strongly overlapping concepts.
    8. Appends valid candidates without modifying existing records.

Important:
    - This script never modifies questions.jsonl.
    - Generated questions are always marked verified=False.
    - AI-suggested evidence must be checked against the original PDF.
    - Unanswerable candidates retain their inspected source paper in
      generation_source_paper_ids, while their gold paper/chunk lists
      remain empty until manually verified.
    - Cross-paper generation remains disabled until related-paper
      selection is implemented.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from generator import QwenGenerator


# ---------------------------------------------------------------------
# Project paths and supported question categories
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "data/evaluation"

CHUNKS_PATH = PROJECT_ROOT / "data/processed/chunks.jsonl"
PAPERS_PATH = PROJECT_ROOT / "data/metadata/papers.json"
QUESTIONS_PATH = EVALUATION_DIR / "questions.jsonl"

DEFAULT_OUTPUT_PATH = EVALUATION_DIR / "candidates.jsonl"

SUPPORTED_CATEGORIES = [
    "paper_specific",
    "methodology",
    "mathematical",
    "results_limitations",
    "cross_paper_comparison",
    "unanswerable",
]

# Cross-paper questions require deliberate related-paper pairing. Do not
# silently generate them from randomly selected, unrelated papers.
IMPLEMENTED_CATEGORIES = {
    "paper_specific",
    "methodology",
    "mathematical",
    "results_limitations",
    "unanswerable",
}

logger = logging.getLogger("benchmark_generator")


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records with line-specific error reporting."""

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, "
                    f"line {line_number}."
                )

            records.append(record)

    return records


def load_corpus() -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    """Load paper metadata and group extracted chunks by arXiv ID."""

    with PAPERS_PATH.open("r", encoding="utf-8") as file:
        papers = json.load(file)

    if not isinstance(papers, list):
        raise ValueError("papers.json must contain a JSON list.")

    paper_map = {
        paper["arxiv_id"]: paper
        for paper in papers
        if isinstance(paper, dict) and paper.get("arxiv_id")
    }

    chunks = load_jsonl(CHUNKS_PATH)
    chunks_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)

    required_fields = {
        "chunk_id",
        "arxiv_id",
        "title",
        "page_number",
        "text",
    }

    for chunk in chunks:
        if not required_fields.issubset(chunk):
            logger.warning(
                "Skipping incomplete chunk: %s",
                chunk.get("chunk_id", "unknown"),
            )
            continue

        if chunk["arxiv_id"] in paper_map:
            chunks_by_paper[chunk["arxiv_id"]].append(chunk)

    for paper_chunks in chunks_by_paper.values():
        paper_chunks.sort(
            key=lambda item: (
                item.get("chunk_index", 0),
                item.get("page_number", 0),
            )
        )

    logger.info(
        "Loaded %d papers and %d usable chunks.",
        len(paper_map),
        sum(len(items) for items in chunks_by_paper.values()),
    )

    return papers, paper_map, chunks_by_paper


def load_all_evaluation_records(
    output_path: Path,
) -> list[dict[str, Any]]:
    """
    Load the benchmark and every existing JSONL candidate file.

    Reading all candidate files prevents a paper used in an earlier
    generation run from being selected again merely because a different
    output filename is supplied for the next run.
    """

    paths = set(EVALUATION_DIR.glob("*.jsonl"))
    paths.add(QUESTIONS_PATH)

    # Include a custom output file if it already exists outside the
    # evaluation directory, or if its path was not found by the glob.
    if output_path.exists():
        paths.add(output_path)

    records: list[dict[str, Any]] = []

    for path in sorted(paths, key=lambda item: str(item)):
        if not path.exists():
            continue

        loaded = load_jsonl(path)
        records.extend(loaded)
        logger.info("Loaded %d records from %s", len(loaded), path.name)

    return records


def extract_represented_paper_ids(
    records: list[dict[str, Any]],
) -> set[str]:
    """
    Collect papers already represented by benchmark or candidate records.

    Answerable records use expected_paper_ids. New unanswerable candidates
    use generation_source_paper_ids to retain provenance without claiming
    those papers as gold evidence.
    """

    represented: set[str] = set()

    for record in records:
        for field in (
            "expected_paper_ids",
            "generation_source_paper_ids",
            "source_paper_ids",
        ):
            values = record.get(field, [])

            if isinstance(values, list):
                represented.update(
                    value for value in values
                    if isinstance(value, str) and value.strip()
                )

    return represented


# ---------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------

def select_relevant_chunks(
    chunks: list[dict[str, Any]],
    paper: dict[str, Any],
    category: str,
    count: int,
) -> list[dict[str, Any]]:
    """
    Select topically relevant and reasonably diverse passages.

    The paper title and abstract provide a topic vocabulary. Candidate
    passages are scored using rare topic-term overlap and category-specific
    keywords. A diversity penalty discourages selecting near-duplicate
    passages.

    This is a lightweight lexical heuristic, not a semantic relevance
    guarantee. All selected evidence must still be manually verified.
    """

    if count <= 0 or not chunks:
        return []

    category_keywords = {
        "paper_specific": {
            "finding", "findings", "contribution", "observation",
            "propose", "proposed", "novel", "demonstrate", "demonstrated",
            "discover", "discovered", "insight", "result",
        },
        "methodology": {
            "method", "methodology", "architecture", "algorithm",
            "framework", "module", "mechanism", "procedure", "training",
            "optimization", "objective", "design", "implementation",
            "network", "model", "pipeline",
        },
        "mathematical": {
            "equation", "equations", "mathematical", "formulation",
            "gradient", "derivative", "loss", "objective", "optimization",
            "theorem", "proof", "constraint", "probability",
            "matrix", "function", "parameter",
        },
        "results_limitations": {
            "result", "results", "experiment", "experiments",
            "evaluation", "accuracy", "performance", "comparison",
            "ablation", "limitation", "limitations", "failure",
            "benchmark", "improvement", "baseline", "metric",
        },
        "unanswerable": set(),
    }

    stopwords = {
        "the", "and", "for", "that", "with", "from", "this", "these",
        "those", "are", "was", "were", "been", "being", "have", "has",
        "had", "into", "their", "there", "which", "while", "where",
        "when", "what", "how", "why", "can", "could", "would", "should",
        "may", "might", "will", "not", "but", "also", "than", "then",
        "such", "using", "used", "use", "based", "our", "we", "they",
        "its", "each", "both", "all", "one", "two", "three", "more",
        "most", "some", "other", "between", "within", "across",
    }

    def tokenize(text: str) -> set[str]:
        """Normalize text into a set of informative lexical terms."""
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
            if token not in stopwords
        }

    topic_text = (
        f"{paper.get('title', '')} {paper.get('abstract', '')}"
    )
    topic_terms = tokenize(topic_text)
    category_terms = category_keywords.get(category, set())

    # Calculate document frequency across this paper's passages. Terms
    # appearing in fewer passages receive more weight in the topic score.
    chunk_terms = [tokenize(chunk.get("text", "")) for chunk in chunks]
    document_frequency: dict[str, int] = {}

    for terms in chunk_terms:
        for term in terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1

    scored_chunks = []

    for chunk, terms in zip(chunks, chunk_terms):
        topic_overlap = terms & topic_terms
        category_overlap = terms & category_terms

        topic_score = sum(
            1.0 / (document_frequency[term] + 1)
            for term in topic_overlap
        ) / max(len(terms) ** 0.5, 1.0)

        category_score = 0.25 * len(category_overlap)

        scored_chunks.append({
            "chunk": chunk,
            "terms": terms,
            "score": topic_score + category_score,
        })

    selected = []
    remaining = scored_chunks.copy()

    # Greedily select passages while penalizing excessive lexical overlap.
    while remaining and len(selected) < min(count, len(chunks)):
        best_item = None
        best_adjusted_score = float("-inf")

        for item in remaining:
            max_similarity = 0.0

            for chosen in selected:
                union = item["terms"] | chosen["terms"]

                if union:
                    similarity = (
                        len(item["terms"] & chosen["terms"]) / len(union)
                    )
                    max_similarity = max(max_similarity, similarity)

            adjusted_score = item["score"] - 0.20 * max_similarity

            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_item = item

        if best_item is None:
            break

        selected.append(best_item)
        remaining.remove(best_item)

    return [item["chunk"] for item in selected]


def select_source_material(
    category: str,
    chunks_by_paper: dict[str, list[dict[str, Any]]],
    paper_map: dict[str, dict[str, Any]],
    rng: random.Random,
    chunks_per_paper: int,
    max_chars_per_chunk: int,
    represented_paper_ids: set[str],
    paper_id: str | None = None,
    allow_reused_papers: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Select one paper and its relevant source passages.

    By default, papers already represented in the benchmark or candidate
    files are excluded. An explicitly supplied paper_id takes precedence
    over that exclusion, allowing deliberate, targeted generation without
    enabling general paper reuse.
    """

    if category == "cross_paper_comparison":
        raise ValueError(
            "Cross-paper generation is disabled until related-paper "
            "selection is implemented."
        )

    eligible_ids = [
        candidate_id
        for candidate_id, paper_chunks in chunks_by_paper.items()
        if candidate_id in paper_map and paper_chunks
    ]

    if paper_id is not None:
        # Explicit targeting is intentional: the caller has selected this
        # paper, so its existing benchmark representation is not a blocker.
        if paper_id not in paper_map:
            raise ValueError(
                f"Requested paper ID is absent from papers.json: {paper_id}"
            )

        if paper_id not in chunks_by_paper or not chunks_by_paper[paper_id]:
            raise ValueError(
                f"No usable extracted chunks found for paper: {paper_id}"
            )

        eligible_ids = [paper_id]

    elif not allow_reused_papers:
        eligible_ids = [
            candidate_id
            for candidate_id in eligible_ids
            if candidate_id not in represented_paper_ids
        ]

    if not eligible_ids:
        if allow_reused_papers:
            raise RuntimeError(
                "No eligible papers with usable chunks were found."
            )

        raise RuntimeError(
            "No unused papers remain. The benchmark/candidate files "
            "already cover every eligible paper. If intentional paper "
            "reuse is needed, rerun with --allow-reused-papers."
        )

    # Randomize only when selecting from multiple eligible papers.
    # Explicit paper targeting must remain deterministic.
    if paper_id is None:
        rng.shuffle(eligible_ids)

    selected_paper_id = eligible_ids[0]
    paper = paper_map[selected_paper_id]

    selected_papers = [{
        "arxiv_id": selected_paper_id,
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", "")[:2000],
    }]

    selected_chunks = select_relevant_chunks(
        chunks=chunks_by_paper[selected_paper_id],
        paper=paper,
        category=category,
        count=chunks_per_paper,
    )

    selected_sources = []

    for index, chunk in enumerate(selected_chunks, start=1):
        selected_sources.append({
            "source_id": f"S{index}",
            "chunk_id": chunk["chunk_id"],
            "arxiv_id": selected_paper_id,
            "title": chunk["title"],
            "page_number": chunk["page_number"],
            "text": chunk["text"][:max_chars_per_chunk],
        })

    if not selected_sources:
        raise RuntimeError(
            f"No source passages selected for paper {selected_paper_id}."
        )

    return selected_papers, selected_sources


# ---------------------------------------------------------------------
# Prompt construction and response parsing
# ---------------------------------------------------------------------

def build_generation_prompt(
    category: str,
    papers: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> str:
    """
    Build a strict evidence-grounded benchmark-generation prompt.

    Abstracts provide topic context only. Supporting source labels must
    refer to passages that directly support the question and its concepts.
    """

    paper_context = []

    for paper in papers:
        paper_context.append(
            f"""
PAPER:
arXiv ID: {paper['arxiv_id']}
Title: {paper['title']}
Abstract (topic context only; not supporting evidence):
{paper['abstract']}
"""
        )

    source_context = []

    for source in sources:
        source_context.append(
            f"""
SOURCE LABEL: {source['source_id']}
Paper ID: {source['arxiv_id']}
Title: {source['title']}
Page: {source['page_number']}
Passage:
{source['text']}
"""
        )

    category_instructions = {
        "paper_specific": (
            "Create a precise question about a specific finding, "
            "observation, contribution or technical detail."
        ),
        "methodology": (
            "Focus on the proposed methodology, model architecture, "
            "algorithm, experimental procedure or design decisions."
        ),
        "mathematical": (
            "Focus on equations, mathematical formulations, optimization "
            "objectives, variables or theoretical assumptions. Only ask "
            "about mathematical details explicitly visible in the passages."
        ),
        "results_limitations": (
            "Focus on experimental results, quantitative comparisons, "
            "limitations, failure cases or conclusions explicitly supported "
            "by the passages."
        ),
        "unanswerable": (
            "Propose a plausible, specific question about a measurement, "
            "deployment setting or experimental detail that may not be "
            "reported. The supplied passages are only a starting point: "
            "they cannot establish absence from the complete paper. Return "
            "an empty source_ids array. A human must inspect the full PDF "
            "before deciding whether the question is truly unanswerable."
        ),
    }

    if category not in category_instructions:
        raise ValueError(f"Unsupported question category: {category}")

    return f"""
You are helping construct a rigorous scientific research benchmark.

Generate exactly ONE candidate question.

QUESTION CATEGORY:
{category}

CATEGORY REQUIREMENT:
{category_instructions[category]}

PAPER INFORMATION:
{"".join(paper_context)}

AVAILABLE EVIDENCE PASSAGES:
{"".join(source_context)}

STRICT EVIDENCE RULES:
1. For answerable questions, the question must be answerable using the
   supplied evidence passages, not merely the paper title or abstract.
2. Every expected concept must be directly supported by at least one
   supplied passage.
3. Select the minimum number of source labels needed to support the
   question and all its expected concepts.
4. Do not include a source merely because it discusses the same general
   research topic.
5. Do not infer missing experimental details or invent numerical results.
6. Do not introduce facts absent from the supplied passages.
7. Do not use the abstract as a substitute for supporting evidence.
8. For unanswerable questions, return no source labels. The question is
   only a hypothesis for human verification against the complete PDF.
9. Never invent source labels.
10. Expected concepts must be concise, independently verifiable facts.

QUESTION QUALITY RULES:
1. Questions must be technically meaningful and specific.
2. Avoid generic questions answerable from common knowledge.
3. Do not simply ask for a paper summary.
4. Prefer questions that require understanding the technical content.
5. Avoid questions that combine unrelated details from different passages.

Return valid JSON only. Do not use Markdown fences or commentary.

OUTPUT FORMAT:
{{
    "question": "The generated question",
    "source_ids": ["S1"],
    "expected_concepts": [
        "First independently verifiable concept",
        "Second independently verifiable concept"
    ]
}}

For unanswerable questions, source_ids must be an empty array.
"""


def parse_model_response(response: str) -> dict[str, Any]:
    """Extract one JSON object from the model response."""

    cleaned = response.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end <= start:
        raise ValueError("Model response does not contain a JSON object.")

    result = json.loads(cleaned[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from the model.")

    return result


# ---------------------------------------------------------------------
# Candidate validation and duplicate detection
# ---------------------------------------------------------------------

def normalize_question(question: str) -> str:
    """Normalize text for exact duplicate-question detection."""

    return re.sub(r"\W+", " ", question.lower()).strip()


CONCEPT_STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "these",
    "those", "are", "was", "were", "been", "being", "have", "has",
    "had", "into", "their", "there", "which", "while", "where",
    "when", "what", "how", "why", "can", "could", "would", "should",
    "may", "might", "will", "not", "but", "also", "than", "then",
    "such", "using", "used", "use", "based", "our", "they", "its",
    "each", "both", "all", "one", "two", "three", "more", "most",
    "some", "other", "between", "within", "across", "paper", "method",
    "proposed", "approach", "model", "result", "results",
}


def concept_tokens(concepts: Any) -> set[str]:
    """Extract informative terms from a record's expected concepts."""

    if not isinstance(concepts, list):
        return set()

    text = " ".join(
        item for item in concepts
        if isinstance(item, str)
    ).lower()

    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text)
        if token not in CONCEPT_STOPWORDS
    }


def concepts_overlap(
    candidate_concepts: Any,
    existing_concepts: Any,
    threshold: float = 0.72,
) -> bool:
    """
    Detect substantial lexical overlap between expected-concept sets.

    This conservative Jaccard check is a guardrail, not semantic proof.
    Human review remains necessary, especially when two papers use
    different terminology for the same underlying idea.
    """

    candidate_terms = concept_tokens(candidate_concepts)
    existing_terms = concept_tokens(existing_concepts)

    if not candidate_terms or not existing_terms:
        return False

    union = candidate_terms | existing_terms
    similarity = len(candidate_terms & existing_terms) / len(union)

    return similarity >= threshold


def is_duplicate_candidate(
    candidate: dict[str, Any],
    existing_records: list[dict[str, Any]],
    existing_questions: set[str],
) -> tuple[bool, str]:
    """Reject exact question duplicates or strongly overlapping concepts."""

    normalized = normalize_question(candidate["question"])

    if normalized in existing_questions:
        return True, "Exact or normalized question duplicate."

    for record in existing_records:
        if concepts_overlap(
            candidate.get("expected_concepts", []),
            record.get("expected_concepts", []),
        ):
            return True, (
                "Expected concepts substantially overlap with "
                f"{record.get('question_id', 'an existing record')}."
            )

    return False, ""


def build_candidate(
    generated: dict[str, Any],
    category: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Validate generated fields and resolve temporary source labels.

    Paper and chunk IDs are derived from selected source records, never
    accepted as arbitrary IDs generated by the language model.
    """

    question = generated.get("question")
    source_ids = generated.get("source_ids", [])
    raw_concepts = generated.get("expected_concepts", [])

    if not isinstance(question, str) or len(question.strip()) < 20:
        raise ValueError("Generated question is missing or too short.")

    if not isinstance(source_ids, list):
        raise ValueError("source_ids must be a list.")

    if not isinstance(raw_concepts, list):
        raise ValueError("expected_concepts must be a list.")

    concepts = [
        concept.strip()
        for concept in raw_concepts
        if isinstance(concept, str) and concept.strip()
    ]

    if not concepts:
        raise ValueError("No valid expected concepts were generated.")

    source_map = {
        source["source_id"]: source
        for source in sources
    }

    if any(
        not isinstance(source_id, str) or source_id not in source_map
        for source_id in source_ids
    ):
        raise ValueError("Model returned an unknown source label.")

    resolved_sources = [
        source_map[source_id]
        for source_id in dict.fromkeys(source_ids)
    ]

    if category == "unanswerable":
        if resolved_sources:
            raise ValueError(
                "Unanswerable candidates must have an empty source_ids list."
            )

        # Preserve the inspected paper for audit and future duplicate
        # prevention, without incorrectly treating it as gold evidence.
        generation_source_paper_ids = list(dict.fromkeys(
            source["arxiv_id"] for source in sources
        ))
        expected_paper_ids: list[str] = []
        expected_chunk_ids: list[str] = []

    else:
        if not resolved_sources:
            raise ValueError(
                "Answerable question has no supporting source passages."
            )

        generation_source_paper_ids = list(dict.fromkeys(
            source["arxiv_id"] for source in resolved_sources
        ))
        expected_paper_ids = generation_source_paper_ids.copy()
        expected_chunk_ids = list(dict.fromkeys(
            source["chunk_id"] for source in resolved_sources
        ))

    source_description = ", ".join(generation_source_paper_ids)

    return {
        "question_id": "",
        "question": question.strip(),
        "question_type": category,
        "answerable": category != "unanswerable",
        "expected_paper_ids": expected_paper_ids,
        "expected_chunk_ids": expected_chunk_ids,
        "expected_concepts": concepts,
        "verified": False,
        "verification_notes": (
            "AI-generated candidate. Verify the question, answerability, "
            "expected concepts and gold evidence against the original PDF. "
            f"Generation source paper(s): {source_description}."
        ),
        "generation_source_paper_ids": generation_source_paper_ids,
    }


# ---------------------------------------------------------------------
# Output management
# ---------------------------------------------------------------------

def next_candidate_id(existing: list[dict[str, Any]]) -> int:
    """Find the next available candidate sequence number across all files."""

    numbers = []

    for record in existing:
        match = re.fullmatch(
            r"cand(\d+)",
            str(record.get("question_id", "")),
        )

        if match:
            numbers.append(int(match.group(1)))

    return max(numbers, default=0) + 1


def append_candidate(path: Path, candidate: dict[str, Any]) -> None:
    """Append one validated candidate without replacing existing records."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(candidate, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Command-line configuration
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line configuration."""

    parser = argparse.ArgumentParser(
        description="Generate draft SciLitRAG benchmark questions."
    )

    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of candidate questions to generate (default: 5).",
    )

    parser.add_argument(
        "--categories",
        type=str,
        default=(
            "paper_specific,methodology,mathematical,"
            "results_limitations,unanswerable"
        ),
        help=(
            "Comma-separated question categories. Cross-paper comparison "
            "is currently disabled."
        ),
    )

    parser.add_argument(
        "--paper-id",
        type=str,
        default=None,
        help=(
            "Generate candidates from this exact arXiv paper ID. "
            "Explicit targeting allows this paper to be reused even if "
            "it already appears in the benchmark or candidate files."
        ),
    )

    parser.add_argument(
        "--chunks-per-paper",
        type=int,
        default=6,
        help="Number of source chunks selected from each paper.",
    )

    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=1400,
        help="Maximum characters included from each chunk.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        help="Maximum Qwen generation tokens per question.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible source selection.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Candidate JSONL output path.",
    )

    parser.add_argument(
        "--allow-reused-papers",
        action="store_true",
        help=(
            "Allow selecting papers already represented in benchmark or "
            "candidate files. Use only after unused-paper coverage is "
            "exhausted or when intentional reuse is desired."
        ),
    )

    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be greater than zero.")

    if args.chunks_per_paper <= 0:
        parser.error("--chunks-per-paper must be greater than zero.")

    if args.max_chars_per_chunk <= 0:
        parser.error("--max-chars-per-chunk must be greater than zero.")

    categories = [
        item.strip()
        for item in args.categories.split(",")
        if item.strip()
    ]

    if not categories:
        parser.error("At least one category must be specified.")

    invalid_categories = set(categories) - set(SUPPORTED_CATEGORIES)

    if invalid_categories:
        parser.error(
            f"Invalid categories: {sorted(invalid_categories)}. "
            f"Supported: {SUPPORTED_CATEGORIES}"
        )

    disabled_categories = set(categories) - IMPLEMENTED_CATEGORIES

    if disabled_categories:
        parser.error(
            "These categories are not implemented yet: "
            f"{sorted(disabled_categories)}. "
            "Cross-paper generation needs related-paper selection."
        )

    args.categories = categories

    # Resolve relative output paths against the project root, not whichever
    # directory happens to be active when the script is invoked.
    if not args.output.is_absolute():
        args.output = PROJECT_ROOT / args.output

    return args


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main() -> None:
    """Generate and append benchmark candidates from unused papers."""

    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    rng = random.Random(args.seed)

    _, paper_map, chunks_by_paper = load_corpus()

    existing_records = load_all_evaluation_records(args.output)
    represented_paper_ids = extract_represented_paper_ids(existing_records)

    existing_questions = {
        normalize_question(record.get("question", ""))
        for record in existing_records
        if isinstance(record.get("question"), str)
    }

    candidate_number = next_candidate_id(existing_records)

    logger.info("Existing benchmark/candidate records: %d", len(existing_records))
    logger.info("Papers already represented: %d", len(represented_paper_ids))
    logger.info(
        "Unused eligible papers: %d",
        sum(
            1 for paper_id, paper_chunks in chunks_by_paper.items()
            if paper_chunks
            and paper_id in paper_map
            and (
                args.allow_reused_papers
                or paper_id not in represented_paper_ids
            )
        ),
    )
    logger.info("Requested new candidates: %d", args.count)
    logger.info("Categories: %s", args.categories)
    logger.info("Target paper: %s", args.paper_id if args.paper_id else "Automatic unused-paper selection")
    logger.info("Output: %s", args.output)

    generator = None
    generated_count = 0
    failed_count = 0

    try:
        # Load Qwen once and reuse it for every candidate in this run.
        generator = QwenGenerator(
            max_new_tokens=args.max_new_tokens
        )

        for index in range(args.count):
            category = args.categories[index % len(args.categories)]

            logger.info(
                "Generating candidate %d/%d | Category: %s",
                index + 1,
                args.count,
                category,
            )

            try:
                papers, sources = select_source_material(
                    category=category,
                    chunks_by_paper=chunks_by_paper,
                    paper_map=paper_map,
                    rng=rng,
                    chunks_per_paper=args.chunks_per_paper,
                    max_chars_per_chunk=args.max_chars_per_chunk,
                    represented_paper_ids=represented_paper_ids,
                    paper_id=args.paper_id,
                    allow_reused_papers=args.allow_reused_papers,
                )

                selected_paper_ids = {
                    paper["arxiv_id"] for paper in papers
                }

                prompt = build_generation_prompt(
                    category=category,
                    papers=papers,
                    sources=sources,
                )

                response = generator.generate(
                    prompt=prompt,
                    system_prompt=(
                        "You generate rigorous scientific evaluation "
                        "questions. Follow the requested JSON schema "
                        "exactly and never invent evidence identifiers."
                    ),
                    max_new_tokens=args.max_new_tokens,
                )

                generated = parse_model_response(response)

                candidate = build_candidate(
                    generated=generated,
                    category=category,
                    sources=sources,
                )

                # Ensure the provenance recorded on the candidate matches
                # the paper actually selected for this generation attempt.
                if set(candidate["generation_source_paper_ids"]) != selected_paper_ids:
                    raise ValueError(
                        "Candidate source-paper provenance does not match "
                        "the selected paper."
                    )

                duplicate, reason = is_duplicate_candidate(
                    candidate=candidate,
                    existing_records=existing_records,
                    existing_questions=existing_questions,
                )

                if duplicate:
                    logger.warning(
                        "Candidate rejected: %s | %s",
                        reason,
                        candidate["question"],
                    )
                    failed_count += 1
                    continue

                candidate["question_id"] = f"cand{candidate_number:03d}"

                append_candidate(args.output, candidate)

                # Update in-memory state immediately so subsequent
                # candidates in the same run cannot reuse this paper or
                # duplicate the newly appended question/concepts.
                existing_records.append(candidate)
                existing_questions.add(
                    normalize_question(candidate["question"])
                )
                represented_paper_ids.update(
                    candidate["generation_source_paper_ids"]
                )

                candidate_number += 1
                generated_count += 1

                logger.info(
                    "Saved %s | Paper: %s | %s",
                    candidate["question_id"],
                    ", ".join(candidate["generation_source_paper_ids"]),
                    candidate["question"],
                )

            except Exception:
                failed_count += 1
                logger.exception(
                    "Candidate generation failed for category: %s",
                    category,
                )

    finally:
        # Release the 14B model and its GPU memory even if generation fails.
        if generator is not None:
            generator.close()

    logger.info("=" * 65)
    logger.info("BENCHMARK GENERATION SUMMARY")
    logger.info("Successfully generated: %d", generated_count)
    logger.info("Failed or rejected: %d", failed_count)
    logger.info("Candidate file: %s", args.output)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()