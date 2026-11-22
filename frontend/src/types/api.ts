
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
 * Retrieval can be unrestricted, scoped to one paper, or scoped to
 * multiple papers for a comparison conversation. The backend rejects
 * requests that specify both single-paper and multi-paper scopes.
 */
export interface AskRequest {
  question: string;
  arxiv_id?: string | null;
  arxiv_ids?: string[] | null;
  history: ConversationHistoryMessage[];
}

/**
 * One evidence passage returned by the backend.
 *
 * Each citation identifies the source paper and the exact passage
 * used to ground the generated answer.
 */
export interface Citation {
  evidence_id: string;
  chunk_id: string;
  arxiv_id: string;
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
