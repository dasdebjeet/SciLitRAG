
/*
 * API types for SciLitRAG.
 *
 * These interfaces mirror the Pydantic request and response schemas
 * defined by the FastAPI backend.
 */

/**
 * A previous message supplied as context for a multi-turn RAG request.
 *
 * Only the role and text content are sent; frontend IDs, timestamps,
 * and assistant citation metadata are not needed by the backend.
 */
export interface ConversationHistoryMessage {
  role: "user" | "assistant";
  content: string;
}

/**
 * Payload submitted to POST /api/v1/ask.
 *
 * Retrieval can be unrestricted, scoped to one indexed paper, scoped
 * to multiple indexed papers, or scoped to one uploaded PDF.
 * The backend rejects requests that specify multiple retrieval scopes.
 */
export interface AskRequest {
  question: string;
  arxiv_id?: string | null;
  arxiv_ids?: string[] | null;
  upload_id?: string | null;
  history: ConversationHistoryMessage[];
}

/**
 * One evidence passage returned by the backend.
 *
 * A citation can originate from the indexed arXiv library or an
 * uploaded research document. Uploaded-document citations do not
 * have an arXiv identifier.
 */
export interface Citation {
  evidence_id: string;
  chunk_id: string;
  source_type: "arxiv" | "upload";
  source_id: string | null;
  arxiv_id: string | null;
  filename: string | null;
  title: string;
  page_number: number;
  text: string;
  reranker_score: number;
}

/**
 * Complete response returned by POST /api/v1/ask.
 */
export interface AskResponse {
  question: string;
  answer: string;
  citations: Citation[];
  context_token_count: number;

  // Total time spent processing the complete API request, including
  // retrieval, reranking, context preparation, and response generation.
  total_response_duration_seconds: number;

  // Time spent inside the language model's generate() call only.
  generation_duration_seconds?: number | null;
  generation_token_count?: number | null;
  tokens_per_second?: number | null;
}

/**
 * Response returned by POST /api/v1/uploads.
 *
 * The upload ID is an opaque identifier used to scope subsequent
 * research questions to this uploaded document.
 */
export interface UploadResponse {
  upload_id: string;
  filename: string;
  title: string;
  page_count: number;
  total_characters: number;
  created_at: string;
}

/**
 * Payload submitted to POST /api/v1/compare.
 *
 * The backend accepts two or three distinct arXiv identifiers.
 * Selection validation is also enforced by the backend.
 */
export interface CompareRequest {
  arxiv_ids: string[];
}

/**
 * Complete response returned by POST /api/v1/compare.
 *
 * The answer contains the generated comparison, while citations
 * retain the evidence passages used to support its conclusions.
 */
export interface CompareResponse {
  arxiv_ids: string[];
  answer: string;
  citations: Citation[];
  context_token_count: number;

  // Time spent inside the language model's generate() call.
  generation_duration_seconds?: number | null;

  // Number of tokens produced by the language model.
  generation_token_count?: number | null;

  // Language-model generation throughput.
  tokens_per_second?: number | null;

  // Total server-side comparison request duration.
  total_response_duration_seconds: number;
}

/**
 * One research paper returned by GET /api/v1/papers.
 *
 * This interface intentionally contains only public API fields.
 * Local filesystem metadata is never exposed to the frontend.
 */
export interface Paper {
  arxiv_id: string;
  title: string;
  authors: string[];
  abstract: string;
  published: string;
  categories: string[];
  primary_category: string;
  pdf_url: string;
}
