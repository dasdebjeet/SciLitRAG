# SciLitRAG — Setup Guide

This guide explains how to configure, initialize, ingest research papers into the local database, and run **SciLitRAG** in a local development environment.

---

## 1. Prerequisites

SciLitRAG combines a Python backend, a React frontend, Qdrant for vector search, and a locally hosted `Qwen3-14B` language model. Ensure the following tools and hardware are available before proceeding:

| Requirement | Purpose |
| :--- | :--- |
| **Windows 11 + WSL2 (Ubuntu)** | Recommended local development environment |
| **Python 3.12** | Backend runtime |
| **`uv`** | Python dependency and virtual environment management |
| **Node.js & npm** | Frontend runtime and package management |
| **Docker Desktop** | Running the Qdrant vector database container |
| **NVIDIA GPU + Drivers** | Hardware acceleration for local embedding and LLM inference |
| **`Qwen3-14B` Model** | Local language model used for answer synthesis |
| **Local Inference Runtime** | Serves `Qwen3-14B` to the backend application |

> **Note:** The specific local inference server and environment configuration depend on your deployment setup. Ensure your inference runtime is configured and reachable before using the question-answering functionality.

---

## 2. Clone the Repository

Clone the repository and navigate into the project root:

```bash
git clone https://github.com/dasdebjeet/SciLitRAG.git
cd SciLitRAG
```

All backend and ingestion commands in this guide should be executed from the repository root unless stated otherwise.

---

## 3. Backend Setup

SciLitRAG uses **`uv`** to manage its virtual environment and lockfile-pinned dependencies.

1. From the project root, install and synchronize dependencies:

   ```bash
   uv sync
   ```

2. Verify that `uv` resolves Python 3.12:

   ```bash
   uv run python --version
   ```

---

## 4. Frontend Setup

Open a separate terminal, navigate to the `frontend` directory, and install Node dependencies:

```bash
cd frontend
npm install
```

---

## 5. Start Qdrant (Vector Database)

Qdrant stores and searches dense passage embeddings for the retrieval pipeline.

1. Launch **Docker Desktop** on Windows and wait until the Docker Engine is active.
2. Open your WSL2 terminal and verify Docker connectivity:

   ```bash
   docker ps
   ```

3. From the project root, start the configured Qdrant service:

   ```bash
   docker compose up -d
   ```

4. Confirm that the container is running:

   ```bash
   docker ps
   ```

5. Qdrant should be reachable at:

   `http://localhost:16333`

---

## 6. Configure the Local Language Model

Before starting the FastAPI backend, ensure **`Qwen3-14B`** is ready to serve inference requests:

1. Confirm that `Qwen3-14B` weights are available to your local inference runtime.
2. Start the local inference server.
3. Verify network reachability from your WSL2 backend environment.
4. Confirm that the model name and inference endpoint URL match your application configuration.

> **Configuration Note:** Reference your actual deployment configuration files for exact runtime startup commands and environment variable names.

The local language model is required for answer generation. It is not required to run the offline paper ingestion stages described below.

---

## 7. Start the Backend Server

From the project root, start the FastAPI development server via `uv`:

```bash
uv run uvicorn src.scilitrag.api.main:app --host 127.0.0.1 --port 8000
```

- **Backend Base URL:** `http://127.0.0.1:8000`
- **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`

Keep this terminal session running while using the application.

---

## 8. Start the Frontend Server

Open a new terminal, navigate to the frontend directory, and start the Vite development server:

```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

- **Frontend URL:** `http://localhost:5173`

Keep this terminal session running alongside the backend.

> **Security Note:** Passing `--host 0.0.0.0` exposes the Vite development server on all network interfaces, allowing the Windows host browser to reach the WSL2 server. Use this binding only in trusted local development networks.

---

## 9. Service Endpoints Overview

Once all services are running, verify access across the stack:

| Service | Local URL | Role |
| :--- | :--- | :--- |
| **SciLitRAG Frontend** | `http://localhost:5173` | Web UI for Q&A, Library, Comparison, and PDF Uploads |
| **FastAPI Backend** | `http://127.0.0.1:8000` | REST API orchestrating retrieval and generation |
| **Swagger UI** | `http://127.0.0.1:8000/docs` | Interactive OpenAPI schema and endpoint testing |
| **Qdrant Vector DB** | `http://localhost:16333` | Dense vector search and passage metadata store |

---

## 10. First-Time Paper Ingestion

A fresh installation does not contain the research-paper corpus or its generated indexes. Run the offline ingestion pipeline to download papers, extract their text, create chunks, generate embeddings, and upload those embeddings to Qdrant.

Run every command below from the **repository root**.

### 10.1. Start Qdrant

Ensure Docker Desktop is running and the Qdrant container is available:

```bash
docker compose up -d
```

### 10.2. Download research papers

Run the arXiv paper-fetching script:

```bash
uv run python scripts/ingestion/fetch_papers.py
```

The default configuration processes up to **5 papers** across these arXiv categories:

- `cs.LG` — Machine Learning
- `cs.AI` — Artificial Intelligence
- `cs.CV` — Computer Vision
- `cs.CL` — Computation and Language

To request a larger corpus, specify the maximum number of results:

```bash
uv run python scripts/ingestion/fetch_papers.py --max-results 120
```

You can also explicitly specify the categories:

```bash
uv run python scripts/ingestion/fetch_papers.py --max-results 150 --categories cs.LG cs.AI cs.CV cs.CL
```

The downloader validates PDF files, retries transient download failures, and avoids downloading papers already present in the local corpus.

**Generated files:**

| Location | Description |
| :--- | :--- |
| `data/raw/` | Downloaded research-paper PDFs |
| `data/metadata/papers.json` | Paper metadata, including arXiv IDs, titles, and PDF paths |

### 10.3. Extract PDF text

Extract page-level text from the downloaded PDFs:

```bash
uv run python scripts/ingestion/extract_text.py
```

The extraction pipeline uses **PyMuPDF** to extract embedded PDF text while preserving page-level metadata and source information.

**Output:**

```text
data/processed/pages.jsonl
```

Each JSONL record represents a PDF page and preserves information such as the paper ID, title, page number, PDF path, and extracted text.

> **Limitation:** This stage extracts embedded PDF text; it does not perform OCR on scanned or image-only pages. Such pages may produce little or no extracted text.

### 10.4. Create text chunks

Split the extracted pages into smaller, token-aware chunks:

```bash
uv run python scripts/ingestion/chunk_pages.py
```

The chunking pipeline uses the **`BAAI/bge-m3` tokenizer** and processes each page independently to preserve accurate page citations.

Default configuration:

| Parameter | Default |
| :--- | :--- |
| Tokenizer | `BAAI/bge-m3` |
| Chunk size | 500 tokens |
| Overlap | 80 tokens |

**Output:**

```text
data/processed/chunks.jsonl
```

Each chunk retains its source-paper metadata and page-level provenance. Chunk IDs are deterministic, making the generated records reproducible.

The chunking parameters can be customized:

```bash
uv run python scripts/ingestion/chunk_pages.py --chunk-size 500 --overlap 80
```

### 10.5. Generate dense embeddings

Generate vector embeddings for the processed chunks:

```bash
uv run python scripts/ingestion/embed_chunks.py
```

The embedding pipeline uses **`BAAI/bge-m3`** through Sentence Transformers.

Default configuration:

| Parameter | Default |
| :--- | :--- |
| Embedding model | `BAAI/bge-m3` |
| Device | `cuda` |
| Batch size | 16 |
| Vector dimensions | 1024 |
| Normalization | Enabled |

**Output:**

```text
data/processed/embeddings.jsonl
```

The embedding pipeline supports incremental checkpointing and validates existing records using chunk IDs and text hashes. If embedding generation is interrupted, completed batches can be reused when the script is run again.

For a smaller smoke test, use the optional limit:

```bash
uv run python scripts/ingestion/embed_chunks.py --limit 10
```

> **Important:** The limited run is intended for testing. Generate embeddings for the complete chunk dataset before performing the full Qdrant ingestion.

If CUDA is unavailable, embedding generation can be run on CPU:

```bash
uv run python scripts/ingestion/embed_chunks.py --device cpu
```

### 10.6. Initialize the Qdrant collection

Before uploading embeddings, initialize or verify the Qdrant collection:

```bash
uv run python scripts/utilities/setup_qdrant.py
```

The script connects to `http://localhost:16333` and creates the collection if it does not already exist.

Collection configuration:

| Setting | Value |
| :--- | :--- |
| Collection | `scilit_rag` |
| Named vector | `dense` |
| Dimensions | 1024 |
| Distance metric | Cosine |

If the collection already exists, the script verifies its configuration without deleting or recreating existing data.

### 10.7. Upload embeddings to Qdrant

After generating the complete embedding dataset and initializing the collection, run:

```bash
uv run python scripts/ingestion/ingest_qdrant.py
```

The ingestion script:

- Validates the embedding records against the source chunks before writing to Qdrant.
- Checks the embedding model, dimensions, normalization, and source-text hashes.
- Uploads points in batches (128 by default).
- Uses deterministic point IDs to make repeated upserts idempotent.
- Preserves chunk text and paper metadata in the Qdrant payload.
- Verifies the final Qdrant point count against the expected number of embeddings.

The script does not create, delete, or recreate the collection. Run the collection setup step first.

A successful run reports the collection name, number of uploaded points, verified point count, vector dimensions, and embedding model.

### 10.8. BM25 lexical index

SciLitRAG maintains its BM25 lexical index separately from Qdrant's dense vector index.

The BM25 retriever loads the processed chunk dataset from:

```text
data/processed/chunks.jsonl
```

It builds an **in-memory BM25Okapi index** when initialized. This index is used alongside Qdrant's dense retrieval results, with Reciprocal Rank Fusion (RRF) combining the two retrieval lists.

There is no separate persistent BM25 database to upload to Qdrant.

**Important:** Keep `chunks.jsonl` available when running the backend. The lexical index is rebuilt in memory when the BM25 retriever starts.

### 10.9. Verify the ingestion outputs

After the pipeline completes, the expected local data layout is:

```text
data/
├── raw/
│   └── *.pdf
├── metadata/
│   └── papers.json
└── processed/
    ├── pages.jsonl
    ├── chunks.jsonl
    └── embeddings.jsonl
```

Verify the Qdrant collection and its point count:

```bash
uv run python scripts/utilities/setup_qdrant.py
```

The output should confirm:

- Collection: `scilit_rag`
- Named vector: `dense`
- Dimensions: `1024`
- Distance: `Cosine`
- Existing points: greater than zero

The ingestion script also performs an exact final point-count check after uploading.

Once the corpus is populated, start or restart the backend so the BM25 retriever can load the processed chunks. The application can then use dense and lexical retrieval together.

### 10.10. Re-running ingestion

The ingestion stages can be run again when adding more papers.

Recommended order:

1. Fetch additional papers.
2. Re-run PDF text extraction.
3. Re-run chunk generation.
4. Re-run embedding generation.
5. Verify the Qdrant collection.
6. Re-run Qdrant ingestion.
7. Restart the backend to rebuild the in-memory BM25 index from the updated chunks.

The downloader avoids duplicate paper downloads, the embedding pipeline validates and reuses compatible checkpoint records, and deterministic Qdrant point IDs allow existing points to be updated rather than duplicated.

> **Data Note:** The `data/raw/` and `data/processed/` directories are generated local corpus data and are not included in the Git repository. Each new installation must generate its own corpus and indexes.

---

## 11. Troubleshooting

### Docker or Qdrant Unavailable

- Confirm **Docker Desktop** is running and the Docker Engine has finished initializing.
- Run `docker ps` to check container status.
- Inspect `compose.yaml` for port mappings.
- Verify that port `16333` is reachable at `http://localhost:16333`.

### Paper Download Fails

- Check your internet connection.
- Retry the downloader; transient network failures are handled with retries.
- Check the terminal output for individual paper download or PDF validation errors.
- Confirm that `data/raw/` and `data/metadata/` are writable.

### PDF Text Extraction Produces No Text

- Confirm that the PDF exists under `data/raw/`.
- Check whether the PDF contains an embedded text layer.
- Scanned or image-only PDFs are not OCR-processed by the current extraction script.

### Chunking Fails

- Confirm that `data/processed/pages.jsonl` exists and contains valid records.
- Ensure the tokenizer can be downloaded or is already cached locally.
- Check the terminal output for page-level processing errors.

### Embedding Generation Fails

- Confirm that `data/processed/chunks.jsonl` exists and contains valid chunks.
- Verify that the `BAAI/bge-m3` model can be loaded.
- For CUDA errors or insufficient GPU memory, try a smaller batch size or CPU execution:

  ```bash
  uv run python scripts/ingestion/embed_chunks.py --batch-size 4
  ```

  ```bash
  uv run python scripts/ingestion/embed_chunks.py --device cpu
  ```

- If the process is interrupted, rerun the script to reuse compatible completed embedding records.

### Qdrant Ingestion Fails

- Confirm that Docker Desktop and the Qdrant container are running.
- Verify that the collection was initialized using `setup_qdrant.py`.
- Confirm that both `chunks.jsonl` and `embeddings.jsonl` exist.
- Check that the embeddings were generated using `BAAI/bge-m3` with 1024 dimensions.
- Review the ingestion logs for validation errors or point-count mismatches.

### Backend Fails to Start

- Verify Python 3.12 is active (`uv run python --version`).
- Re-run `uv sync` to ensure all dependencies are installed cleanly.
- Check if port `8000` is already bound by another process.
- Inspect terminal logs for missing configuration or import errors.
- Confirm that the processed chunk dataset is available for BM25 initialization.

### Frontend Fails to Start

- Confirm Node.js and `npm` are installed and accessible in WSL2.
- Re-run `npm install` inside `frontend/`.
- Check whether port `5173` is already in use.

### Local Language Model Unreachable

- Confirm the local inference runtime process is active.
- Verify `Qwen3-14B` is loaded and matches the configured model identifier.
- Test connectivity to the inference endpoint from within WSL2.

### No Papers Returned During Retrieval

- Verify Qdrant is running and accessible on port `16333`.
- Confirm the `scilit_rag` collection exists and contains points.
- Confirm `data/processed/chunks.jsonl` exists for BM25 retrieval.
- Restart the backend after updating the processed chunk dataset.
- Check backend logs for Qdrant connection or retrieval errors.

---

## 12. Related Documentation

| Document | Description |
| :--- | :--- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | High-level system architecture and component layers |
| [RAG_PIPELINE.md](RAG_PIPELINE.md) | Offline ingestion and online retrieval/generation pipeline |
| [API_REFERENCE.md](API_REFERENCE.md) | REST endpoints, Swagger UI discovery, and local server commands |
| [EVALUATION.md](EVALUATION.md) | Retrieval, context-efficiency, and answer-quality benchmarks |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Local development workflows and contributor guidelines |

---

*SciLitRAG — A locally hosted research-paper assistant powered by hybrid retrieval and large language models.*