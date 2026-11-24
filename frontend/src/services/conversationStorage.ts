
import type {
  Conversation,
  ConversationUploadedDocument,
} from "../types/conversation";

/**
 * Versioned storage key allows the persisted conversation format
 * to evolve without silently interpreting older data incorrectly.
 *
 * Version 1 remains valid because the new `papers` property is optional
 * and existing single-paper fields are retained.
 */
const STORAGE_KEY = "scilitrag.conversations.v1";

interface ConversationStorage {
  version: 1;
  conversations: Conversation[];
}

/**
 * Validates a paper snapshot before accepting persisted conversation data.
 *
 * Both fields are required because paper references must remain meaningful
 * when a conversation is restored independently of the Paper Library.
 */
function isConversationPaper(value: unknown): boolean {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const paper = value as Record<string, unknown>;

  return (
    typeof paper.arxivId === "string" &&
    paper.arxivId.trim().length > 0 &&
    typeof paper.paperTitle === "string" &&
    paper.paperTitle.trim().length > 0
  );
}

/**
 * Validates uploaded-document metadata before accepting persisted data.
 *
 * The backend upload ID is required to retrieve the correct document.
 * Display metadata is also validated to prevent malformed composer state.
 */
function isConversationUploadedDocument(
  value: unknown,
): value is ConversationUploadedDocument {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const document = value as Record<string, unknown>;

  return (
    typeof document.uploadId === "string" &&
    document.uploadId.trim().length > 0 &&
    typeof document.filename === "string" &&
    document.filename.trim().length > 0 &&
    typeof document.title === "string" &&
    typeof document.pageCount === "number" &&
    Number.isInteger(document.pageCount) &&
    document.pageCount > 0
  );
}

/**
 * Performs basic runtime validation before trusting data loaded from
 * localStorage. TypeScript types alone cannot validate parsed JSON.
 *
 * The optional `papers` property preserves compatibility with conversations
 * saved before multi-paper research was introduced.
 */
function isConversation(value: unknown): value is Conversation {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const conversation = value as Record<string, unknown>;

  if (
    typeof conversation.id !== "string" ||
    typeof conversation.title !== "string" ||
    !Array.isArray(conversation.messages) ||
    (conversation.arxivId !== null &&
      typeof conversation.arxivId !== "string") ||
    (conversation.paperTitle !== null &&
      typeof conversation.paperTitle !== "string") ||
    typeof conversation.createdAt !== "string" ||
    typeof conversation.updatedAt !== "string"
  ) {
    return false;
  }

  // Validate uploaded-document metadata only when the property is present.
  if (
    conversation.uploadedDocument !== undefined &&
    !isConversationUploadedDocument(conversation.uploadedDocument)
  ) {
    return false;
  }

  // Validate multi-paper metadata only when the property is present.
  if (
    conversation.papers !== undefined &&
    (!Array.isArray(conversation.papers) ||
      !conversation.papers.every(isConversationPaper))
  ) {
    return false;
  }

  return conversation.messages.every((message: unknown) => {
    if (typeof message !== "object" || message === null) {
      return false;
    }

    const item = message as Record<string, unknown>;

    if (
      typeof item.id !== "string" ||
      typeof item.content !== "string" ||
      typeof item.createdAt !== "string"
    ) {
      return false;
    }

    if (item.role === "user") {
      return true;
    }

    return (
      item.role === "assistant" &&
      Array.isArray(item.citations) &&
      typeof item.contextTokenCount === "number"
    );
  });
}

/**
 * Loads conversations from browser storage.
 *
 * Invalid or unavailable storage returns an empty list rather than
 * preventing the application from starting.
 */
export function loadConversations(): Conversation[] {
  if (typeof window === "undefined") {
    return [];
  }

  try {
    const storedValue = window.localStorage.getItem(STORAGE_KEY);

    if (!storedValue) {
      return [];
    }

    const parsed: unknown = JSON.parse(storedValue);

    if (
      typeof parsed !== "object" ||
      parsed === null ||
      !("version" in parsed) ||
      !("conversations" in parsed)
    ) {
      return [];
    }

    const storage = parsed as ConversationStorage;

    if (
      storage.version !== 1 ||
      !Array.isArray(storage.conversations) ||
      !storage.conversations.every(isConversation)
    ) {
      return [];
    }

    return [...storage.conversations].sort(
      (first, second) =>
        Date.parse(second.updatedAt) - Date.parse(first.updatedAt),
    );
  } catch (error) {
    console.error("Failed to load saved conversations:", error);
    return [];
  }
}

/**
 * Persists the complete conversation collection.
 *
 * Errors are allowed to propagate so the calling hook can inform the
 * UI if browser storage is unavailable or its quota has been exceeded.
 */
export function saveConversations(conversations: Conversation[]): void {
  if (typeof window === "undefined") {
    return;
  }

  const storage: ConversationStorage = {
    version: 1,
    conversations,
  };

  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(storage));
}
