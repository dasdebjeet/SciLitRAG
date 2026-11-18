import { useCallback, useEffect, useMemo, useState } from "react";

import type { ChatMessage, Conversation } from "../types/conversation";
import {
  loadConversations,
  saveConversations,
} from "../services/conversationStorage";

/**
 * Generates a unique identifier for conversations and messages.
 * The fallback supports environments where randomUUID is unavailable.
 */
function generateId(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

/**
 * Creates a readable conversation title from its first user message.
 */
function generateConversationTitle(question: string): string {
  const normalized = question.trim().replace(/\s+/g, " ");

  if (normalized.length <= 48) {
    return normalized || "New chat";
  }

  return `${normalized.slice(0, 48).trimEnd()}...`;
}

/**
 * Keeps the most recently updated conversation at the top of the list.
 */
function sortConversations(
  conversations: Conversation[],
): Conversation[] {
  return [...conversations].sort(
    (first, second) =>
      Date.parse(second.updatedAt) - Date.parse(first.updatedAt),
  );
}

/**
 * Manages conversation state and browser persistence.
 *
 * Conversation messages and metadata are persisted locally. This hook
 * does not send previous messages to the backend; each research request
 * remains independent until backend chat-history support is implemented.
 */
export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>(
    () => loadConversations(),
  );

  const [activeConversationId, setActiveConversationId] = useState<
    string | null
  >(null);

  const [storageError, setStorageError] = useState<string | null>(null);

  /** Resolve the active conversation from the current conversation list. */
  const activeConversation = useMemo(
    () =>
      conversations.find(
        (conversation) => conversation.id === activeConversationId,
      ) ?? null,
    [conversations, activeConversationId],
  );

  /**
   * Persist whenever the conversation collection changes.
   * A storage failure is exposed to the UI instead of silently ignored.
   */
  useEffect(() => {
    try {
      saveConversations(conversations);
      setStorageError(null);
    } catch (error) {
      console.error("Failed to save conversations:", error);
      setStorageError(
        "Unable to save conversations in this browser. Check available storage.",
      );
    }
  }, [conversations]);

    /**
     * Creates and immediately activates a conversation.
     * Called when the user sends the first message, not when they click New Chat.
     * Paper metadata is captured as a snapshot for future restoration.
     */
    const createConversation = useCallback(
      (arxivId: string | null = null, paperTitle: string | null = null) => {
        const now = new Date().toISOString();

        const conversation: Conversation = {
          id: generateId(),
          title: "New chat",
          messages: [],
          arxivId,
          paperTitle,
          createdAt: now,
          updatedAt: now,
        };

        setConversations((previous) =>
          sortConversations([conversation, ...previous]),
        );
        setActiveConversationId(conversation.id);

        return conversation;
      },
      [],
    );

  /**
   * Activates an existing conversation without modifying its messages.
   * Conversation IDs are supplied by the conversation list, so this
   * callback only updates the active selection and leaves the collection
   * untouched.
   */
  const selectConversation = useCallback(
    (conversationId: string | null) => {
      setActiveConversationId(conversationId);
    },
    [],
  );

  /**
   * Appends a message to a conversation and updates its activity time.
   * The first user message automatically becomes the conversation title.
   */
  const addMessage = useCallback(
    (conversationId: string, message: ChatMessage) => {
      setConversations((previous) =>
        sortConversations(
          previous.map((conversation) => {
            if (conversation.id !== conversationId) {
              return conversation;
            }

            const isFirstUserMessage =
              message.role === "user" &&
              !conversation.messages.some(
                (existingMessage) => existingMessage.role === "user",
              );

            return {
              ...conversation,
              title: isFirstUserMessage
                ? generateConversationTitle(message.content)
                : conversation.title,
              messages: [...conversation.messages, message],
              updatedAt: new Date().toISOString(),
            };
          }),
        ),
      );
    },
    [],
  );

  /** Updates a conversation title without changing its messages. */
  const renameConversation = useCallback(
    (conversationId: string, title: string) => {
      const normalizedTitle = title.trim();

      if (!normalizedTitle) {
        return;
      }

      setConversations((previous) =>
        sortConversations(
          previous.map((conversation) =>
            conversation.id === conversationId
              ? {
                  ...conversation,
                  title: normalizedTitle,
                  updatedAt: new Date().toISOString(),
                }
              : conversation,
          ),
        ),
      );
    },
    [],
  );

  /** Updates the paper scope associated with a conversation. */
  const updatePaperScope = useCallback(
    (
      conversationId: string,
      arxivId: string | null,
      paperTitle: string | null,
    ) => {
      setConversations((previous) =>
        sortConversations(
          previous.map((conversation) =>
            conversation.id === conversationId
              ? {
                  ...conversation,
                  arxivId,
                  paperTitle,
                  updatedAt: new Date().toISOString(),
                }
              : conversation,
          ),
        ),
      );
    },
    [],
  );

  /** Deletes a conversation and clears the active selection if necessary. */
  const deleteConversation = useCallback((conversationId: string) => {
    setConversations((previous) =>
      previous.filter((conversation) => conversation.id !== conversationId),
    );

    setActiveConversationId((previous) =>
      previous === conversationId ? null : previous,
    );
  }, []);

  return {
    conversations,
    activeConversationId,
    activeConversation,
    storageError,
    createConversation,
    selectConversation,
    addMessage,
    renameConversation,
    updatePaperScope,
    deleteConversation,
  };
}
