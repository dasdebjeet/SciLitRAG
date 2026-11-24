# SciLitRAG — API Reference

## 1. Introduction

**SciLitRAG** exposes its backend functionality through a REST API built with **FastAPI**, connecting the React frontend to the underlying research retrieval, document management, and local language model generation workflows.

* **Base URL (Local Development):** `http://127.0.0.1:8000`
* **API Version Prefix:** Research and upload routes are versioned under `/api/v1`.

---

## 2. API Overview

The application exposes **six endpoints** organized across three functional groups:

| Group | Method | Endpoint | Description |
| :--- | :--- | :--- | :--- |
| **Health** | `GET` | `/health` | Check API service liveness |
| **Health** | `GET` | `/health/ready` | Check application and dependency readiness |
| **Research** | `POST` | `/api/v1/ask` | Answer a research question within the active scope |
| **Research** | `GET` | `/api/v1/papers` | List available indexed research papers |
| **Research** | `POST` | `/api/v1/compare` | Compare multiple indexed research papers |
| **Uploads** | `POST` | `/api/v1/uploads` | Upload a research PDF for conversation-scoped querying |

---

## 3. Health Endpoints

Health endpoints expose service availability and readiness status for monitoring and local startup verification.

### 3.1 Check API Liveness

* **Endpoint:** `GET /health`
* **Purpose:** Verifies that the FastAPI server process is running and accepting HTTP requests.

```bash
curl http://127.0.0.1:8000/health
```

### 3.2 Check Application Readiness

* **Endpoint:** `GET /health/ready`
* **Purpose:** Verifies application readiness independently from the lightweight liveness check.

```bash
curl http://127.0.0.1:8000/health/ready
```

---

## 4. Research Endpoints

Research endpoints power single-paper Q&A, multi-turn conversations, indexed-paper discovery, and cross-paper comparison.

### 4.1 Ask a Research Question

* **Endpoint:** `POST /api/v1/ask`
* **Purpose:** Submits a research question to the SciLitRAG retrieval and generation pipeline, returning an evidence-backed response scoped to the active document selection.

#### Supported Workflows

* Single-turn research question answering
* Multi-turn conversational follow-up questions
* Queries scoped to indexed research papers
* Queries scoped to an uploaded research PDF

#### Request Lifecycle

1. Receive the research question, conversation history, and document scope.
2. Resolve the active document scope (uploaded document vs. indexed paper collection).
3. Execute hybrid retrieval, cross-encoder reranking, and adaptive context construction.
4. Generate a grounded response via the local `Qwen3-14B` runtime.
5. Return the generated answer along with source references and supporting excerpts.

> *Schema Note: Consult the live OpenAPI specification at `/docs` or `/openapi.json` for exact request body fields and response models.*

---

### 4.2 List Research Papers

* **Endpoint:** `GET /api/v1/papers`
* **Purpose:** Retrieves metadata for the indexed research paper collection to populate the frontend Paper Library and paper-selection controls.

```bash
curl http://127.0.0.1:8000/api/v1/papers
```

> *Schema Note: Exact response payload fields are defined by the backend response model in `/openapi.json`.*

---

### 4.3 Compare Research Papers

* **Endpoint:** `POST /api/v1/compare`
* **Purpose:** Performs comparative question answering across multiple user-selected indexed papers.

#### Typical Use Cases

* Comparing methodologies and experimental setups across studies
* Identifying areas of consensus, contradiction, or complementary findings
* Contrasting quantitative results or problem formulations

#### Request Lifecycle

1. Receive the comparison prompt and selected paper identifiers.
2. Resolve the multi-paper retrieval scope.
3. Retrieve and assemble cross-paper evidence context.
4. Generate a comparative synthesis using the local language model.
5. Return the comparative response and source citations to the frontend.

> *Schema Note: Inspect request and response schemas interactively in Swagger UI (`/docs`).*

---

## 5. Upload Endpoints

### 5.1 Upload a Research Paper

* **Endpoint:** `POST /api/v1/uploads`
* **Purpose:** Accepts a research-paper PDF so it can be bound to a conversation and queried as the active document scope.

#### Upload Constraints

| Property | Constraint |
| :--- | :--- |
| **Supported Format** | `PDF` |
| **Maximum File Size** | `25 MB` |

> *Schema Note: Exact multipart form field names, accepted MIME types, and response schemas should be referenced directly from `/openapi.json`.*

---

## 6. Interactive Schema Discovery

FastAPI automatically generates an OpenAPI 3 specification from the application's route and Pydantic definitions. During local development, the following interactive documentation interfaces are exposed:

| Interface | URL | Purpose |
| :--- | :--- | :--- |
| **Swagger UI** | `http://127.0.0.1:8000/docs` | Interactive API explorer and live request testing |
| **ReDoc** | `http://127.0.0.1:8000/redoc` | Clean, readable API reference documentation |
| **OpenAPI JSON** | `http://127.0.0.1:8000/openapi.json` | Machine-readable OpenAPI contract (source of truth) |

### Testing Endpoints via Swagger UI

1. Navigate to `http://127.0.0.1:8000/docs`.
2. Expand the target endpoint card.
3. Click **Try it out**.
4. Populate required query parameters, multipart files, or JSON request bodies.
5. Click **Execute** and inspect the HTTP status code and response payload.

---

## 7. Related Documentation

| Guide | Use it for |
| :--- | :--- |
| [Project README](../README.md) | Product overview and quick start |
| [System architecture](ARCHITECTURE.md) | Components and request flows |
| [RAG pipeline](RAG_PIPELINE.md) | Ingestion, retrieval, and generation details |
| [Evaluation](EVALUATION.md) | Benchmark methodology and results |
| [Setup](SETUP.md) | Installing dependencies and starting local services |
| [Development](DEVELOPMENT.md) | Contributor workflows and debugging |

---

## 8. Source of Truth

This guide summarizes endpoint purpose and usage. The live OpenAPI document at `/openapi.json` is authoritative for request and response schemas, status codes, and validation rules.
