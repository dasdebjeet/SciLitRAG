import type { Citation } from "./api";

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
 * An assistant response with its associated research evidence.
 *
 * Keeping citations alongside the response ensures that conversation
 * switching restores the correct sources for each answer.
 */
export interface AssistantMessage {
  id: string;
  role: "assistant";
  content: string;
  citations: Citation[];
  contextTokenCount: number;
  createdAt: string;

  // Persist generation statistics with the response so they survive
  // conversation switching and restoration from local storage.
  generationDurationSeconds?: number;
  generationTokenCount?: number;
  tokensPerSecond?: number;
}

/** A message exchanged within a conversation. */
export type ChatMessage = UserMessage | AssistantMessage;

/**
 * A persistent research conversation.
 *
 * Paper metadata is stored as a snapshot to preserve the original
 * research scope independently of future library changes.
 */
export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  arxivId: string | null;
  paperTitle: string | null;
  createdAt: string;
  updatedAt: string;
}
