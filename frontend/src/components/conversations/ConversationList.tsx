
import { useState, type FormEvent, type KeyboardEvent } from "react";
import { Check, Pencil, Trash2, X } from "lucide-react";

import "./ConversationList.css";
import type { Conversation } from "../../types/conversation";

interface ConversationListProps {
  conversations: Conversation[];
  onSelectConversation: (conversationId: string) => void;
  onRenameConversation: (conversationId: string, title: string) => void;
  onDeleteConversation: (conversationId: string) => void;
}

/** Groups conversations by their most recent activity date. */
type ConversationGroup = {
  label: string;
  conversations: Conversation[];
};

/**
 * Returns the calendar-day difference between a date and today.
 * Local midnight normalization avoids timezone-related grouping errors.
 */
function getCalendarDayDifference(date: Date): number {
  const today = new Date();

  const todayMidnight = new Date(
    today.getFullYear(),
    today.getMonth(),
    today.getDate(),
  );

  const dateMidnight = new Date(
    date.getFullYear(),
    date.getMonth(),
    date.getDate(),
  );

  return Math.floor(
    (todayMidnight.getTime() - dateMidnight.getTime()) /
      (24 * 60 * 60 * 1000),
  );
}

/** Formats the exact activity time shown beside the conversation metadata. */
function formatActivityDate(timestamp: string): string {
  const date = new Date(timestamp);

  if (Number.isNaN(date.getTime())) {
    return "";
  }

  const dayDifference = getCalendarDayDifference(date);

  const time = date.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });

  if (dayDifference === 0) {
    return time;
  }

  if (dayDifference === 1) {
    return `Yesterday, ${time}`;
  }

  if (dayDifference <= 7) {
    return date.toLocaleDateString([], {
      weekday: "long",
      hour: "numeric",
      minute: "2-digit",
    });
  }

  return date.toLocaleDateString([], {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

/** Describes how recently a conversation was updated. */
function formatRelativeActivity(timestamp: string): string {
  const date = new Date(timestamp);

  if (Number.isNaN(date.getTime())) {
    return "Activity unavailable";
  }

  const elapsedMilliseconds = Math.max(0, Date.now() - date.getTime());
  const elapsedMinutes = Math.floor(elapsedMilliseconds / 60_000);

  if (elapsedMinutes < 1) {
    return "Updated just now";
  }

  if (elapsedMinutes < 60) {
    return `Updated ${elapsedMinutes} min ago`;
  }

  const elapsedHours = Math.floor(elapsedMinutes / 60);

  if (elapsedHours < 24) {
    return `Updated ${elapsedHours} hr ago`;
  }

  const elapsedDays = Math.floor(elapsedHours / 24);

  if (elapsedDays === 1) {
    return "Updated yesterday";
  }

  return `Updated ${elapsedDays} days ago`;
}

/**
 * Organizes conversations into date-based sections.
 * The conversation hook is expected to sort them newest-first.
 */
function groupConversations(
  conversations: Conversation[],
): ConversationGroup[] {
  const groups = new Map<string, Conversation[]>();

  for (const conversation of conversations) {
    const date = new Date(conversation.updatedAt);

    const dayDifference = Number.isNaN(date.getTime())
      ? Number.POSITIVE_INFINITY
      : getCalendarDayDifference(date);

    const label =
      dayDifference <= 0
        ? "Today"
        : dayDifference === 1
          ? "Yesterday"
          : dayDifference <= 7
            ? "Previous 7 days"
            : "Older";

    const existing = groups.get(label) ?? [];
    existing.push(conversation);
    groups.set(label, existing);
  }

  const sectionOrder = [
    "Today",
    "Yesterday",
    "Previous 7 days",
    "Older",
  ];

  return sectionOrder
    .filter((label) => groups.has(label))
    .map((label) => ({
      label,
      conversations: groups.get(label)!,
    }));
}

function ConversationList({
  conversations,
  onSelectConversation,
  onRenameConversation,
  onDeleteConversation,
}: ConversationListProps) {
  const groups = groupConversations(conversations);

  // Tracks the conversation currently being renamed.
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editedTitle, setEditedTitle] = useState("");

  // Tracks the conversation awaiting deletion confirmation.
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  /** Start editing and close any pending delete confirmation. */
  function handleStartRename(conversation: Conversation) {
    setEditingId(conversation.id);
    setEditedTitle(conversation.title);
    setConfirmDeleteId(null);
  }

  /** Save a valid title and exit editing mode. */
  function handleSaveRename(
    event: FormEvent<HTMLFormElement>,
    conversationId: string,
  ) {
    event.preventDefault();

    const normalizedTitle = editedTitle.trim();

    if (!normalizedTitle) {
      return;
    }

    onRenameConversation(conversationId, normalizedTitle);
    setEditingId(null);
    setEditedTitle("");
  }

  /** Cancel editing without changing the stored title. */
  function handleCancelRename() {
    setEditingId(null);
    setEditedTitle("");
  }

  /** Allow Escape to cancel the rename operation. */
  function handleRenameKeyDown(
    event: KeyboardEvent<HTMLInputElement>,
  ) {
    if (event.key === "Escape") {
      event.preventDefault();
      handleCancelRename();
    }
  }

  /** Delete only after the user explicitly confirms the action. */
  function handleConfirmDelete(conversationId: string) {
    onDeleteConversation(conversationId);
    setConfirmDeleteId(null);
  }

  if (conversations.length === 0) {
    return (
      <section className="conversation-history">
        <header className="conversation-history-header">
          <h2>Your conversations</h2>
          <p>Your research history will appear here.</p>
        </header>

        <div className="conversation-empty-state">
          <h3>No conversations yet</h3>
          <p>
            Start a new research chat to build your conversation history.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section
      className="conversation-history"
      aria-labelledby="conversation-history-title"
    >
      <header className="conversation-history-header">
        <h2 id="conversation-history-title">Your conversations</h2>
        <p>
          {conversations.length} saved{" "}
          {conversations.length === 1 ? "conversation" : "conversations"}
        </p>
      </header>

      <div className="conversation-groups">
        {groups.map((group) => (
          <section
            className="conversation-group"
            key={group.label}
            aria-label={group.label}
          >
            <h3 className="conversation-group-title">{group.label}</h3>

            <div className="conversation-items">
              {group.conversations.map((conversation) => {
                const isEditing = editingId === conversation.id;
                const isConfirmingDelete =
                  confirmDeleteId === conversation.id;

                const messageCount = conversation.messages.length;

                return (
                  <article
                    className="conversation-item"
                    key={conversation.id}
                  >
                    {/* Left side: title and all conversation metadata. */}
                    <div className="conversation-item-main">
                      {isEditing ? (
                        <form
                          className="conversation-rename-form"
                          onSubmit={(event) =>
                            handleSaveRename(event, conversation.id)
                          }
                        >
                          <input
                            className="conversation-rename-input"
                            type="text"
                            value={editedTitle}
                            onChange={(event) =>
                              setEditedTitle(event.target.value)
                            }
                            onKeyDown={handleRenameKeyDown}
                            aria-label="Conversation title"
                            maxLength={120}
                            autoFocus
                          />
                        </form>
                      ) : (
                        <button
                          type="button"
                          className="conversation-item-select"
                          onClick={() =>
                            onSelectConversation(conversation.id)
                          }
                          title={conversation.title || "New chat"}
                        >
                          <span className="conversation-item-title">
                            {conversation.title || "New chat"}
                          </span>

                          <span className="conversation-item-meta">
                            <span>
                              {messageCount}{" "}
                              {messageCount === 1 ? "message" : "messages"}
                            </span>
                            <span aria-hidden="true">·</span>
                            <span>
                              {formatRelativeActivity(conversation.updatedAt)}
                            </span>
                            <span aria-hidden="true">·</span>
                            <time dateTime={conversation.updatedAt}>
                              {formatActivityDate(conversation.updatedAt)}
                            </time>
                          </span>
                        </button>
                      )}
                    </div>

                    {/* Right side: fixed action area prevents row resizing. */}
                    <div className="conversation-item-actions">
                      {isEditing ? (
                        <>
                          <button
                            type="button"
                            className="conversation-action-button"
                            onClick={(event) => {
                              const form = event.currentTarget
                                .closest(".conversation-item")
                                ?.querySelector("form");

                              form?.requestSubmit();
                            }}
                            disabled={!editedTitle.trim()}
                            aria-label="Save conversation title"
                            title="Save"
                          >
                            <Check size={16} />
                          </button>

                          <button
                            type="button"
                            className="conversation-action-button"
                            onClick={handleCancelRename}
                            aria-label="Cancel renaming"
                            title="Cancel"
                          >
                            <X size={16} />
                          </button>
                        </>
                      ) : isConfirmingDelete ? (
                        <div
                          className="conversation-delete-confirm"
                          role="group"
                          aria-label="Confirm conversation deletion"
                        >
                          <span className="conversation-delete-prompt">
                            Delete this conversation?
                          </span>

                          <button
                            type="button"
                            className="conversation-confirm-delete"
                            onClick={() =>
                              handleConfirmDelete(conversation.id)
                            }
                          >
                            Yes
                          </button>

                          <button
                            type="button"
                            className="conversation-cancel-delete"
                            onClick={() => setConfirmDeleteId(null)}
                          >
                            No
                          </button>
                        </div>
                      ) : (
                        <>
                          <button
                            type="button"
                            className="conversation-action-button"
                            onClick={() => handleStartRename(conversation)}
                            aria-label={`Rename ${conversation.title}`}
                            title="Rename conversation"
                          >
                            <Pencil size={15} />
                            <span>Rename</span>
                          </button>

                          <button
                            type="button"
                            className="conversation-action-button conversation-delete-button"
                            onClick={() => {
                              setConfirmDeleteId(conversation.id);
                              setEditingId(null);
                            }}
                            aria-label={`Delete ${conversation.title}`}
                            title="Delete conversation"
                          >
                            <Trash2 size={15} />
                            <span>Delete</span>
                          </button>
                        </>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          </section>
        ))}
      </div>
    </section>
  );
}

export default ConversationList;
