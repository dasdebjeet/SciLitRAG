# SciLitRAG — Setup Guide

This guide explains how to configure, initialize, and run **SciLitRAG** in a local development environment.

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
| **NVIDIA GPU + Drivers** | Hardware acceleration for local LLM inference |
| **`Qwen3-14B` Model** | Local language model used for answer synthesis |
| **Local Inference Runtime** | Serves `Qwen3-14B` to the backend application |

> *Note: The specific local inference server and environment configuration depend on your deployment setup. Ensure your inference runtime is configured and reachable before starting the backend.*

---

## 2. Clone the Repository

Clone the repository and navigate into the project root:

```bash
git clone https://github.com/dasdebjeet/SciLitRAG.git
cd SciLitRAG
```

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
3. If the Qdrant container is not running, inspect `compose.yaml` and start the configured service from the project root:
   ```bash
   docker compose up -d
   ```
4. Confirm the container is up and verify the Qdrant endpoint at **`http://localhost:16333`**:
   ```bash
   docker ps
   ```

---

## 6. Configure the Local Language Model

Before starting the FastAPI backend, ensure **`Qwen3-14B`** is ready to serve inference requests:

1. Confirm that `Qwen3-14B` weights are available to your local inference runtime.
2. Start the local inference server.
3. Verify network reachability from your WSL2 backend environment.
4. Confirm that the model name and inference endpoint URL match your application configuration.

> *Configuration Note: Reference your actual deployment configuration files for exact runtime startup commands and environment variable names.*

---

## 7. Start the Backend Server

From the project root, start the FastAPI development server via `uv`:

```bash
uv run uvicorn src.scilitrag.api.main:app --host 127.0.0.1 --port 8000
```

* **Backend Base URL:** `http://127.0.0.1:8000`
* **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`

Keep this terminal session running while using the application.

---

## 8. Start the Frontend Server

Open a new terminal, navigate to the frontend directory, and start the Vite development server:

```bash
cd /mnt/d/SciLitRAG/frontend
npm run dev -- --host 0.0.0.0
```

* **Frontend URL:** `http://localhost:5173`

Keep this terminal session running alongside the backend.

> **Security Note:** Passing `--host 0.0.0.0` exposes the Vite dev server on all network interfaces (allowing Windows host browsers to reach the WSL2 server). Use this binding only in trusted local development networks.

---

## 9. Service Endpoints Overview

Once all services are running, verify access across the stack:

| Service | Local URL | Role |
| :--- | :--- | :--- |
| **SciLitRAG Frontend** | `http://localhost:5173` | Web UI for Q&A, Library, Comparison, and PDF Uploads |
| **FastAPI Backend** | `http://127.0.0.1:8000` | REST API orchestrating retrieval and generation |
| **Swagger UI** | `http://127.0.0.1:8000/docs` | Interactive OpenAPI schema and endpoint testing |
| **Qdrant Vector DB** | `http://localhost:16333` | Vector similarity search and passage metadata store |

---

## 10. First-Time Data Ingestion

On a fresh installation, the indexed research-paper collection must be downloaded, extracted, embedded, and indexed into Qdrant (and prepared for BM25) before paper search and Q&A will return results.

* See [RAG_PIPELINE.md](RAG_PIPELINE.md) for the end-to-end ingestion architecture.
* Run the ingestion scripts provided in the repository to populate your local Qdrant collection and lexical index.

---

## 11. Troubleshooting

### Docker or Qdrant Unavailable
* Confirm **Docker Desktop** is running and the Docker Engine has finished initializing.
* Run `docker ps` to check container status and inspect `compose.yaml` for port mappings.
* Verify that port `16333` is reachable at `http://localhost:16333`.

### Backend Fails to Start
* Verify Python 3.12 is active (`uv run python --version`).
* Re-run `uv sync` to ensure all dependencies are installed cleanly.
* Check if port `8000` is already bound by another process.
* Inspect terminal logs for missing environment variables or import errors.

### Frontend Fails to Start
* Confirm Node.js and `npm` are installed and accessible in WSL2.
* Re-run `npm install` inside `frontend/`.
* Check whether port `5173` is already in use.

### Local Language Model Unreachable
* Confirm the local inference runtime process is active.
* Verify `Qwen3-14B` is loaded and matches the configured model identifier.
* Test connectivity to the inference endpoint from within WSL2.

### No Papers Returned During Retrieval
* Verify Qdrant is running and accessible on port `16333`.
* Confirm the target vector collection exists and has been populated via the ingestion pipeline.
* Check backend logs for Qdrant connection or retrieval errors.

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