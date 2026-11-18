
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
 * arxiv_id is optional because the backend supports unrestricted
 * retrieval when no paper identifier is provided.
 */
export interface AskRequest {
  question: string;
  arxiv_id?: string | null;
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
