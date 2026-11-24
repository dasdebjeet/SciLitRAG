
import type { Citation } from "./api";

/**
 * A snapshot of a paper associated with a research conversation.
 *
 * Paper metadata is stored locally rather than resolved from the Paper
 * Library so conversations remain restorable if library entries change.
 */
export interface ConversationPaper {
  arxivId: string;
  paperTitle: string;
}

/**
 * Metadata for a PDF uploaded specifically to a research conversation.
 *
 * The PDF itself remains on the backend. Only its identifier and display
 * metadata are persisted in browser storage.
 */
export interface ConversationUploadedDocument {
  uploadId: string;
  filename: string;
  title: string;
  pageCount: number;
}

/**
 * A message submitted by the user.
 *
 * Evidence belongs to the assistant response, not the user message.
 */
export interface UserMessage {
  id: string;
  role: "user";
  content: string;
  createdAt: string;
}

/**
 * An assistant response with its associated research evidence and timing.
 *
 * Keeping citations and performance statistics alongside the response
 * ensures they survive conversation switching and local-storage restoration.
 */
export interface AssistantMessage {
  id: string;
  role: "assistant";
  content: string;
  citations: Citation[];
  contextTokenCount: number;
  createdAt: string;

  // Total time spent processing the complete API request.
  totalResponseDurationSeconds?: number;

  // Language-model generation statistics.
  generationDurationSeconds?: number;
  generationTokenCount?: number;
  tokensPerSecond?: number;
}

/** A message exchanged within a conversation. */
export type ChatMessage = UserMessage | AssistantMessage;

/**
 * A persistent research conversation.
 *
 * The legacy single-paper fields remain available for backward compatibility.
 * The optional papers array supports multi-paper research and comparisons.
 * Paper metadata is stored as a snapshot, independent of future library changes.
 */
export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];

  // Legacy single-paper metadata; retained for existing conversations.
  arxivId: string | null;
  paperTitle: string | null;

  // Optional uploaded-PDF scope, independent of the global Paper Library.
  uploadedDocument?: ConversationUploadedDocument;

  // Complete research scope for multi-paper conversations.
  papers?: ConversationPaper[];

  createdAt: string;
  updatedAt: string;
}
