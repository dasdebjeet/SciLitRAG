# SciLitRAG

**A research-paper assistant powered by hybrid retrieval, cross-encoder reranking, and locally hosted language models.**

SciLitRAG is a retrieval-augmented generation (RAG) system designed to help researchers search, explore, and synthesize information from scientific literature. It retrieves relevant passages from research papers, reranks them by relevance, and generates answers grounded in retrieved evidence with source attribution.

## Key Features

- **Research corpus ingestion:** Fetch and process research papers from arXiv.
- **Semantic search:** Retrieve relevant passages using dense vector embeddings.
- **Keyword search:** Use BM25 to capture exact terminology and technical keywords.
- **Hybrid retrieval:** Combine keyword and semantic retrieval using Reciprocal Rank Fusion (RRF).
- **Cross-encoder reranking:** Improve passage relevance before generation.
- **Evidence-grounded generation:** Generate research answers using a locally hosted language model.
- **Evaluation:** Benchmark retrieval performance, context selection, and generated answer quality.

## Architecture

```text
              Research Papers (arXiv)
                       |
                       v
                 PDF Extraction
                       |
                       v
                  Text Chunking
                       |
             +---------+---------+
             |                   |
             v                   v
         BM25 Index       Dense Embeddings
             |                   |
             |                Qdrant
             |                   |
             +---------+---------+
                       |
                       v
                Hybrid Retrieval
                       |
                       v
              Reciprocal Rank Fusion
                       |
                       v
              Cross-Encoder Reranking
                       |
                       v
                 Context Builder
                       |
                       v
               Local LLM (Qwen3-14B)
                       |
                       v
              Evidence-Grounded Answer
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Dependency management | uv |
| PDF processing | PyMuPDF |
| Embeddings | BAAI/bge-m3 |
| Keyword retrieval | BM25 |
| Vector database | Qdrant |
| Retrieval fusion | Reciprocal Rank Fusion |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Language model | Qwen3-14B |
| Deep learning | PyTorch, Transformers |
| GPU acceleration | CUDA |

## Project Structure

```text
SciLitRAG/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── metadata/
│   └── evaluation/
│
├── src/
│   └── scilitrag/
│       ├── core/
│       │   └── paths.py
│       ├── retrieval/
│       │   ├── bm25_retriever.py
│       │   ├── dense_retriever.py
│       │   ├── hybrid_retriever.py
│       │   └── reranker.py
│       ├── generation/
│       │   └── generator.py
│       └── pipeline/
│           ├── answer_pipeline.py
│           └── context_builder.py
│
├── scripts/
│   ├── ingestion/
│   │   ├── fetch_papers.py
│   │   ├── extract_text.py
│   │   ├── chunk_pages.py
│   │   ├── embed_chunks.py
│   │   └── ingest_qdrant.py
│   ├── evaluation/
│   │   ├── generate_benchmark.py
│   │   ├── evaluate_retrieval.py
│   │   ├── evaluate_context_selection.py
│   │   └── evaluate_answer_quality.py
│   ├── utilities/
│   │   ├── validate_papers.py
│   │   ├── fix_paper_metadata.py
│   │   ├── setup_qdrant.py
│   │   ├── update_qdrant_metadata.py
│   │   └── verify_candidate_questions.py
│   └── diagnostics/
│       ├── test_context_builder.py
│       └── test_qdrant_search.py
│
├── tests/
├── compose.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

**Design principle:**
- `src/scilitrag/` contains reusable application logic.
- `scripts/` contains executable workflows, evaluation tools, utilities, and diagnostics.
- `data/` stores the research corpus and evaluation artifacts.

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker and Docker Compose
- NVIDIA GPU with compatible CUDA support (recommended for GPU inference)

### 1. Clone the repository

```bash
git clone https://github.com/dasdebjeet/SciLitRAG.git
cd SciLitRAG
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Start Qdrant

```bash
docker compose up -d
```

Default Qdrant endpoint:

```text
http://localhost:16333
```

### 4. Prepare the research corpus

Run the ingestion workflow in order:

```bash
uv run python scripts/ingestion/fetch_papers.py
uv run python scripts/ingestion/extract_text.py
uv run python scripts/ingestion/chunk_pages.py
uv run python scripts/ingestion/embed_chunks.py
uv run python scripts/utilities/setup_qdrant.py
uv run python scripts/ingestion/ingest_qdrant.py
```

The scripts support additional configuration through command-line arguments. Use `--help` where available to inspect options.

**Note:** The corpus, generated embeddings, Qdrant collection, and local language-model weights must be prepared or made available separately. `uv sync` installs Python dependencies; it does not download the complete research corpus or model weights.

## Running SciLitRAG

Ask a research question using the complete retrieval and generation pipeline:

```bash
uv run scilitrag-answer --query "How do reversible neural networks improve image reconstruction?"
```

Optional parameters:

```bash
uv run scilitrag-answer \
    --query "Your research question" \
    --retrieval-k 20 \
    --top-k 5 \
    --max-context-tokens 6000 \
    --max-new-tokens 1200 \
    --device cuda
```

Individual retrieval components are also available:

```bash
uv run scilitrag-bm25 --help
uv run scilitrag-dense --help
uv run scilitrag-hybrid --help
uv run scilitrag-rerank --help
```

## Evaluation

SciLitRAG has been evaluated using a manually verified benchmark of 15 research questions, including 13 answerable and 2 unanswerable queries.

### Retrieval Evaluation

| Metric | Hybrid | Reranked |
|---|---:|---:|
| Paper Recall@5 | 1.000 | 1.000 |
| Paper Recall@10 | 1.000 | 1.000 |
| Paper MRR@10 | 1.000 | 1.000 |
| Paper nDCG@5 | 1.000 | 1.000 |
| Chunk Recall@10 | 0.7086 | 0.7560 |
| Chunk MRR@10 | 0.6564 | 0.7564 |
| Chunk nDCG@5 | 0.5510 | 0.5771 |

The reranker improves chunk-level ranking quality, although individual questions still exhibit retrieval weaknesses.

### Context Selection

An adaptive context-selection strategy was evaluated against a fixed top-10 context policy.

- Average context tokens: 3,502 versus 4,711.
- Average selected passages: 7.54 versus 10.
- Approximately 25.7% fewer context tokens.

These results reflect the current benchmark and do not establish adaptive selection as universally optimal.

### Answer Quality

An initial manual quality review was performed on eight generated answers across four questions and two context-selection policies. The outputs were generally sound, but broader evaluation is needed before drawing strong conclusions.

## Current Limitations

- The benchmark is relatively small and requires broader evaluation.
- Retrieval quality varies across individual research questions.
- The current system uses a local language model, with hardware-dependent inference performance.
- The API and interactive user interface are planned extensions.

## Roadmap

- [ ] FastAPI backend for serving the RAG pipeline.
- [ ] React-based research assistant interface.
- [ ] Interactive source and evidence exploration.
- [ ] Cross-paper comparison and synthesis.
- [ ] Expanded retrieval and answer-quality benchmarks.
- [ ] Improved observability and production deployment.

## Project Status

The core ingestion, hybrid retrieval, reranking, generation, and initial evaluation workflows have been implemented. The project is transitioning toward a service-oriented backend and interactive research interface.

---

**SciLitRAG — Making scientific literature easier to search, understand, and connect.**