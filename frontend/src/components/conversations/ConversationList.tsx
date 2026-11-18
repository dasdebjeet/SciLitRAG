
import "./ConversationList.css";
import type { Conversation } from "../../types/conversation";

interface ConversationListProps {
  conversations: Conversation[];
  onSelectConversation: (conversationId: string) => void;
}

/** Groups conversations by their most recent activity date. */
type ConversationGroup = {
  label: string;
  conversations: Conversation[];
};

/**
 * Returns the calendar-day difference between a date and today.
 * Normalizing to local midnight avoids incorrect grouping around midnight.
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

/** Formats the activity timestamp for compact display. */
function formatActivityDate(timestamp: string): string {
  const date = new Date(timestamp);

  if (Number.isNaN(date.getTime())) {
    return "";
  }

  const dayDifference = getCalendarDayDifference(date);

  if (dayDifference === 0) {
    return date.toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit",
    });
  }

  if (dayDifference === 1) {
    return "Yesterday";
  }

  return date.toLocaleDateString([], {
    day: "numeric",
    month: "short",
  });
}

/**
 * Organizes conversations into date-based sections.
 * The input is expected to be sorted newest-first by the conversation hook.
 */
function groupConversations(
  conversations: Conversation[],
): ConversationGroup[] {
  const groups = new Map<string, Conversation[]>();

  for (const conversation of conversations) {
    const date = new Date(conversation.updatedAt);

    // Invalid timestamps are safely placed in the Older section.
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

  // Keep the sections in a predictable chronological order.
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
}: ConversationListProps) {
  const groups = groupConversations(conversations);

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
            <h3 className="conversation-group-title">
              {group.label}
            </h3>

            <div className="conversation-items">
              {group.conversations.map((conversation) => (
                <button
                  type="button"
                  className="conversation-item"
                  key={conversation.id}
                  onClick={() =>
                    onSelectConversation(conversation.id)
                  }
                >
                  <span className="conversation-item-content">
                    <span className="conversation-item-title">
                      {conversation.title || "New chat"}
                    </span>

                    <span className="conversation-item-meta">
                      {conversation.messages.length}{" "}
                      {conversation.messages.length === 1
                        ? "message"
                        : "messages"}
                    </span>
                  </span>

                  <time
                    className="conversation-item-date"
                    dateTime={conversation.updatedAt}
                  >
                    {formatActivityDate(conversation.updatedAt)}
                  </time>
                </button>
              ))}
            </div>
          </section>
        ))}
      </div>
    </section>
  );
}

export default ConversationList;
