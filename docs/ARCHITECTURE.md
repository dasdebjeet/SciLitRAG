# SciLitRAG — System Architecture

**SciLitRAG** is a research-paper question-answering application built around a **Retrieval-Augmented Generation (RAG)** pipeline. It pairs a React frontend and FastAPI backend with hybrid retrieval, cross-encoder reranking, and a locally hosted language model to deliver grounded, citation-backed answers from scientific literature.

### Supported Workflows

- **Single & Multi-Paper Q&A:** Ask targeted or comparative questions across indexed research papers.
- **Document Upload & Q&A:** Upload research PDFs (up to **25 MB**) and query their contents within a conversation.

---

## 1. High-Level Architecture

The system is organized into four logical layers with clear separation of concerns:

| Layer | Core Responsibilities | Key Components |
| :--- | :--- | :--- |
| **Presentation** | Renders UI, manages conversation state and document scope, displays answers and citations | React, TypeScript, Vite |
| **Application** | Exposes REST APIs, validates requests/uploads, orchestrates research workflows | Python 3.12, FastAPI, Uvicorn |
| **Retrieval & Generation** | Retrieves candidate passages, fuses/reranks results, builds context, generates answers | BM25, `BAAI/bge-m3`, RRF, `BAAI/bge-reranker-v2-m3`, `Qwen3-14B` |
| **Infrastructure** | Persists vector embeddings and metadata, hosts local LLM runtime and paper corpus | Qdrant, Local LLM Runtime, Paper Data |

### System Overview Diagram

```mermaid
flowchart TB
    USER["User"]

    subgraph PRESENTATION["Presentation Layer"]
        FE["React + TypeScript + Vite"]
        CHAT["Conversations"]
        LIB["Paper Library"]
        COMPARE_UI["Paper Comparison"]
        UPLOAD_UI["PDF Upload"]
    end

    subgraph APPLICATION["Application Layer"]
        API["FastAPI Application"]
        HEALTH["Health Endpoints"]
        RESEARCH["Research Endpoints"]
        UPLOAD_API["Upload Endpoint"]
    end

    subgraph RAG["Retrieval & Generation Layer"]
        PIPELINE["Answer Pipeline"]
        BM25["BM25 Retriever"]
        DENSE["Dense Retriever"]
        FUSION["Reciprocal Rank Fusion"]
        RERANK["Cross-Encoder Reranker"]
        CONTEXT["Context Builder"]
        GENERATOR["LLM Generator"]
    end

    subgraph INFRA["Infrastructure Layer"]
        QDRANT[("Qdrant")]
        MODEL["Local Qwen3-14B Runtime"]
        DATA["Research Paper Data"]
    end

    USER --> FE
    FE --> CHAT & LIB & COMPARE_UI & UPLOAD_UI
    FE <--> API

    API --> HEALTH & RESEARCH & UPLOAD_API
    RESEARCH --> PIPELINE

    PIPELINE --> BM25 & DENSE
    DENSE <--> QDRANT
    BM25 --> FUSION
    DENSE --> FUSION

    FUSION --> RERANK --> CONTEXT --> GENERATOR
    GENERATOR <--> MODEL
    DATA --> QDRANT
```

> *Note: This diagram illustrates logical component boundaries and primary data flows rather than low-level deployment or network topology.*

---

## 2. Presentation Layer

Built with **React**, **TypeScript**, and **Vite**, the frontend handles user interactions, client-side state, and rendering of responses alongside source citations.

### Interface Modules

| Component | Responsibility |
| :--- | :--- |
| **Conversations** | Create, switch between, and manage research threads |
| **Chat Interface** | Submit prompts and render streaming or completed responses |
| **Paper Library** | Browse available indexed research papers |
| **Paper Selection** | Scope queries to a single indexed paper |
| **Paper Comparison** | Select multiple indexed papers for side-by-side/cross-paper analysis |
| **PDF Upload** | Upload research PDFs (up to **25 MB**) and bind them to a conversation |
| **Source References** | Surface cited passages, source links, and supporting excerpts |
| **Theme Preferences** | Switch and persist UI appearance settings |

### State, History & Scope Rules

- **Persistence & History Window:** Conversation threads are stored in browser `localStorage`. For multi-turn context, the client sends up to the **12 most recent messages**, truncating each message to **2,000 characters**.
- **Scope Precedence:** A query can target a single indexed paper, multiple indexed papers, or a conversation-scoped uploaded PDF. When an **uploaded PDF** is present for the active conversation, it takes precedence over indexed-paper selection.

---

## 3. Application Layer

Implemented in **Python 3.12** with **FastAPI** and served via **Uvicorn**, the backend exposes REST endpoints and coordinates retrieval and generation workflows.

### API Surface

The backend exposes health checks, paper discovery, question answering, paper comparison, and PDF upload routes. See [API_REFERENCE.md](API_REFERENCE.md) for the endpoint list and live schema discovery.

### Request Lifecycle

1. **Dispatch:** The frontend sends an HTTP request with the prompt, conversation history, and active document scope.
2. **Scope Resolution:** FastAPI resolves whether the request targets uploaded-document context or indexed papers.
3. **Workflow Routing:** The corresponding endpoint invokes the Q&A, comparison, or upload workflow.
4. **RAG Execution:** The pipeline retrieves, fuses, reranks, and synthesizes an evidence-backed response.
5. **Response & Render:** FastAPI returns the answer and source metadata, which the frontend renders with citations.

---

## 4. Retrieval & Generation Layer

The RAG pipeline combines lexical matching, dense semantic search, rank fusion, cross-encoder reranking, budget-aware context assembly, and local generation.

1. **Sparse Retrieval (BM25):** Matches exact lexical terms, acronyms, symbols, and domain-specific phrasing across indexed passages.
2. **Dense Retrieval (`BAAI/bge-m3`):** Embeds queries and passages into dense vectors and queries **Qdrant** for semantic similarity.
3. **Hybrid Fusion (Reciprocal Rank Fusion):** Merges BM25 and dense result lists using rank positions (`RRF`) rather than raw score magnitudes, balancing exact keyword precision with semantic recall.
4. **Cross-Encoder Reranking (`BAAI/bge-reranker-v2-m3`):** Jointly scores `(query, passage)` pairs to re-order top candidates with higher cross-attention accuracy than bi-encoder similarity alone.
5. **Context Construction:** Selects passages within the token/context budget, comparing **adaptive context selection** against **fixed top-k** passage selection.
6. **Answer Generation (`Qwen3-14B`):** Synthesizes a grounded response from the assembled evidence and conversation history, including references to supporting passages and source papers where available.

---

## 5. Infrastructure Layer

- **Vector Database (Qdrant):** Stores passage embeddings and payload metadata, serving dense similarity queries. In local development, Qdrant runs via Docker Desktop and is exposed at `http://localhost:16333`.
- **Local LLM Runtime (`Qwen3-14B`):** Runs inference locally rather than calling external third-party LLM APIs. Runtime and hardware/server tuning details are maintained outside this document.
- **Research Corpus & Ingestion:** Source papers and metadata feed the indexing pipeline documented in [RAG_PIPELINE.md](RAG_PIPELINE.md).

---

## 6. Core Workflows

### 6.1 Research Question Answering

```mermaid
sequenceDiagram
    actor User
    participant UI as React Frontend
    participant API as FastAPI
    participant RAG as RAG Pipeline
    participant DB as Qdrant
    participant LLM as Local Qwen3-14B

    User->>UI: Submit research question
    UI->>API: POST /api/v1/ask
    API->>RAG: Process question & scope
    RAG->>RAG: BM25 sparse retrieval
    RAG->>DB: Dense vector search
    DB-->>RAG: Candidate passages & metadata
    RAG->>RAG: Reciprocal Rank Fusion (RRF)
    RAG->>RAG: Cross-encoder reranking
    RAG->>RAG: Construct generation context
    RAG->>LLM: Generate grounded answer
    LLM-->>RAG: Generated response
    RAG-->>API: Answer + source citations
    API-->>UI: JSON response
    UI-->>User: Render answer & references
```

### 6.2 Research Paper Comparison

```mermaid
flowchart TD
    A["Select multiple papers"] --> B["Submit comparison request"]
    B --> C["POST /api/v1/compare"]
    C --> D["Comparison workflow"]
    D --> E["Retrieve cross-paper evidence"]
    E --> F["Build comparative context"]
    F --> G["Local Qwen3-14B"]
    G --> H["Synthesize comparison"]
    H --> I["Render response & citations in UI"]
```

### 6.3 PDF Upload & Querying

```mermaid
flowchart TD
    A["Select PDF (≤ 25 MB)"] --> B["Validate file in frontend"]
    B --> C["POST /api/v1/uploads"]
    C --> D["Backend upload & association"]
    D --> E["Bind PDF to active conversation"]
    E --> F["Ask question in chat"]
    F --> G["Run RAG response workflow"]
    G --> H["Display answer & sources"]
```

> *Internal ingestion, chunking, and lifecycle details for uploaded PDFs are maintained alongside the backend implementation and in [RAG_PIPELINE.md](RAG_PIPELINE.md).*

---

## 7. Local Development Topology

During local development, services run independently across local ports:

| Service | Stack / Runtime | Local Endpoint | Notes |
| :--- | :--- | :--- | :--- |
| **Frontend** | React, TypeScript, Vite | `http://localhost:5173` | Dev server |
| **Backend API** | Python 3.12, FastAPI, Uvicorn | `http://127.0.0.1:8000` | REST API |
| **Vector Store** | Qdrant (Docker Desktop) | `http://localhost:16333` | Dense vector index |
| **Local LLM** | `Qwen3-14B` runtime | *Configured locally* | See local runtime config |