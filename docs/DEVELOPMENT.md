# SciLitRAG — Development Guide

This document provides guidance for developers working on **SciLitRAG**, covering repository layout, local development workflows, testing, debugging, and recommendations for extending the system.

> *For first-time installation and environment setup instructions, see [SETUP.md](SETUP.md).*

---

## 1. Architecture & Stack Overview

SciLitRAG is organized across a React frontend, a FastAPI backend, a hybrid retrieval and reranking pipeline, and local infrastructure services:

| Component | Technology | Responsibility |
| :--- | :--- | :--- |
| **Frontend** | React, TypeScript, Vite | User interface, conversation state, and citation rendering |
| **Backend** | Python 3.12, FastAPI, Uvicorn | REST API endpoints and workflow orchestration |
| **Dense Retrieval** | `BAAI/bge-m3` | Semantic vector embedding and similarity retrieval |
| **Sparse Retrieval** | `BM25` | Lexical keyword-based passage retrieval |
| **Rank Fusion** | Reciprocal Rank Fusion (`RRF`) | Merging sparse and dense candidate rankings |
| **Reranking** | `BAAI/bge-reranker-v2-m3` | Cross-encoder scoring for candidate passage precision |
| **Vector Database** | Qdrant | Embedding storage, metadata payloads, and vector search |
| **Language Model** | `Qwen3-14B` (Local) | Evidence-grounded answer generation |
| **PDF Processing** | `PyMuPDF` | Text extraction from research PDFs |

*For system design, see [ARCHITECTURE.md](ARCHITECTURE.md); for retrieval mechanics, see [RAG_PIPELINE.md](RAG_PIPELINE.md).*

---

## 2. Repository Structure

```text
SciLitRAG/
├── docs/
│   ├── images/
│   ├── ARCHITECTURE.md
│   ├── RAG_PIPELINE.md
│   ├── EVALUATION.md
│   ├── API_REFERENCE.md
│   ├── SETUP.md
│   └── DEVELOPMENT.md
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── ...
├── scripts/
├── src/
│   └── scilitrag/
│       └── api/
│           └── main.py
├── data/
├── tests/
├── compose.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

### Directory Breakdown

* **`src/`**: Core Python backend package. The FastAPI application entry point lives at `src/scilitrag/api/main.py`.
* **`frontend/`**: Independent React + TypeScript single-page application bundled with Vite.
* **`scripts/`**: Supporting operational scripts for data preparation, paper ingestion, and benchmark evaluation.
* **`data/`**: Local development data, including downloaded raw PDFs and intermediate/processed artifacts.
* **`tests/`**: Automated test suite validating backend and pipeline behavior.
* **`docs/`**: Project documentation suite covering architecture, RAG pipeline, evaluation, setup, and API reference.

> *Note: This tree provides a high-level orientation rather than an exhaustive file manifest. Inspect the repository directly for exact module layouts.*

---

## 3. Local Runtime

Use [SETUP.md](SETUP.md) for prerequisites, dependency installation, service startup commands, and local endpoints. Keep the backend, frontend, and required infrastructure in separate terminal sessions while developing.

---

## 4. Working on the Retrieval & Generation Pipeline

The core pipeline executes six sequential stages:

1. **Sparse Retrieval (`BM25`)**
2. **Dense Retrieval (`BAAI/bge-m3`)**
3. **Rank Fusion (Reciprocal Rank Fusion)**
4. **Cross-Encoder Reranking (`BAAI/bge-reranker-v2-m3`)**
5. **Adaptive Context Selection**
6. **Context Construction & Answer Generation (`Qwen3-14B`)**

### Downstream Coupling Considerations

When modifying any stage, account for ripple effects downstream:

* **Chunking changes** impact both `BM25` lexical statistics and `BAAI/bge-m3` embeddings (requiring re-indexing).
* **Embedding model changes** require regenerating and re-indexing vectors in Qdrant.
* **Fusion parameter adjustments** alter the candidate pool passed to the cross-encoder reranker.
* **Reranker modifications** change the priority order of evidence supplied to context selection.
* **Context-selection tuning** directly governs prompt token budget and evidence coverage.
* **Prompt template edits** affect citation formatting, grounding strictness, and abstention behavior.

> **Best Practice:** Evaluate retrieval changes and generation changes independently whenever possible so retrieval regressions are not masked by LLM behavior (and vice versa).

---

## 5. Testing & Validation

### Automated Tests

Run the backend test suite from the project root:

```bash
uv run pytest
```

*(Ensure any required services, environment variables, or test fixtures are available before running integration tests.)*

For frontend verification, inspect available npm scripts in `frontend/package.json`:

```bash
cd frontend
npm run
```

### Retrieval Benchmark Metrics

Assess retrieval modifications against the evaluation benchmark (see [EVALUATION.md](EVALUATION.md)) across both paper and passage granularities:

* **Paper-level:** `Recall@K`, `MRR@K`, `nDCG@K`
* **Chunk-level:** `Recall@K`, `MRR@K`, `nDCG@K`

### Answer Quality Audit Criteria

Because strong retrieval metrics do not guarantee faithful synthesis, manually inspect generated responses for:

* **Factual consistency** with retrieved passages
* **Direct relevance** to the user's question
* **Citation accuracy** and supporting excerpt alignment
* **Appropriate abstention** when questions cannot be answered from available evidence

---

## 6. Debugging Guidelines

### Symptom Triage Matrix

| Symptom | Primary Areas to Investigate |
| :--- | :--- |
| **Frontend fails to load** | Vite dev server output, `npm` dependencies, browser DevTools console |
| **API request fails** | FastAPI terminal logs, route paths, Pydantic request validation errors |
| **Qdrant connection error** | Docker Engine status, `docker ps`, `compose.yaml` port mapping (`16333`) |
| **No relevant papers retrieved** | Qdrant collection existence/count, `BM25` index state, dense search scores |
| **Poor passage ranking** | `RRF` fusion weights/cutoffs, candidate pool size, cross-encoder reranker output |
| **Weak or unsupported answers** | Retrieved chunks, adaptive context selection threshold, system/task prompt |
| **Model inference failure** | Local LLM server logs, model availability, configured inference endpoint |
| **PDF processing failure** | Source PDF integrity, `PyMuPDF` extraction output, upload validation logs |

### Recommended Debugging Sequence

1. **Reproduce** the issue using a minimal, deterministic prompt or API request.
2. **Inspect logs** in both frontend and backend terminals.
3. **Verify dependencies** (Qdrant container health and local `Qwen3-14B` endpoint).
4. **Trace intermediate payloads** passed between retrieval, fusion, reranking, and context assembly.
5. **Isolate the first failing stage** before modifying code.
6. **Apply a single, focused change** (avoid altering multiple pipeline stages simultaneously).
7. **Re-run tests and benchmarks** to confirm the fix without introducing regressions.

---

## 7. Extending SciLitRAG

### Retrieval & Indexing

* Tune `BM25` hyperparameters and tokenization rules.
* Experiment with dense/sparse candidate pool sizes prior to `RRF` fusion.
* Test alternative chunk sizes, overlap windows, or section-aware splitting.
* Benchmark alternative embedding or cross-encoder reranking models against [EVALUATION.md](EVALUATION.md).

### Context Construction

* Refine adaptive passage selection thresholds to reduce redundant chunks while preserving multi-hop evidence.
* Enrich context headers with structured paper metadata (title, section, arXiv ID).
* Evaluate token budget tradeoffs across single-paper vs. multi-paper comparison workflows.

### Generation & Prompting

* Iterate on system prompts to strengthen citation grounding and reduce unsupported extrapolations.
* Standardize refusal behavior for unanswerable queries.
* Optimize multi-turn history summarization or truncation windows.

### Frontend & UX

* Enhance conversation persistence, search, and export capabilities.
* Improve Paper Library filtering and multi-paper comparison views.
* Surface richer inline citation previews alongside loading, empty, and error states.
* Validate all frontend data models against the FastAPI OpenAPI schema.

---

## 8. API Development Workflow

Use Swagger UI (`http://127.0.0.1:8000/docs`) during endpoint development. Before modifying an existing route:

1. Inspect the current route handler and Pydantic schemas in `src/scilitrag/api/`.
2. Identify frontend callers and expected response shapes.
3. Preserve backward compatibility where possible.
4. Test both happy-path and error/validation-failure payloads.
5. Update [API_REFERENCE.md](API_REFERENCE.md) if routes or behaviors change.

---

## 9. Engineering Best Practices

* **Enforce Separation of Concerns:** Keep UI rendering, HTTP routing, retrieval/reranking logic, and LLM client calls cleanly decoupled.
* **Preserve Reproducibility:** Commit updates to `pyproject.toml`, `uv.lock`, and `frontend/package-lock.json` whenever dependencies change.
* **Validate Inputs Early:** Catch corrupt PDFs and malformed requests at system boundaries before expensive embedding or inference steps.
* **Measure Quantitative Impact:** Validate retrieval changes against the benchmark rather than relying solely on ad-hoc prompts.
* **Monitor GPU Memory Budgets:** Embedding (`BAAI/bge-m3`), reranking (`BAAI/bge-reranker-v2-m3`), and generation (`Qwen3-14B`) share local compute resources—monitor VRAM contention.
* **Handle External Failures Gracefully:** Return clear, actionable HTTP errors if Qdrant or the local LLM runtime is unreachable.
* **Protect Environment Secrets:** Never commit local credentials, private endpoints, or machine-specific paths/secrets.
* **Keep Docs Synchronized:** Update `docs/` whenever architecture, setup steps, or API contracts evolve.

---

## 10. Related Documentation

| Document | Description |
| :--- | :--- |
| [Project README](../README.md) | Project overview and quick start |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System layers, components, and data flows |
| [RAG_PIPELINE.md](RAG_PIPELINE.md) | Offline ingestion and online retrieval/generation pipeline |
| [API_REFERENCE.md](API_REFERENCE.md) | REST API endpoints and OpenAPI discovery |
| [SETUP.md](SETUP.md) | Local installation and service startup guide |
| [EVALUATION.md](EVALUATION.md) | Benchmark methodology, metrics, and limitations |

*SciLitRAG — A research-paper assistant built around hybrid retrieval, reranking, adaptive context selection, and locally hosted language models.*