
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { BookOpen, X, ArrowRight } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";

// KaTeX styles are required to display formatted mathematical expressions.
import "katex/dist/katex.min.css";

import ConversationList from "./components/conversations/ConversationList";
import Sidebar from "./components/Sidebar";
import PaperLibrary from "./components/PaperLibrary";

import { useConversations } from "./hooks/useConversations";
import type {
  AssistantMessage,
  UserMessage,
} from "./types/conversation";

import { askQuestion } from "./services/api";
import type { Citation, Paper } from "./types/api";

import "./App.css";

type AppView = "chat" | "library" | "conversations";

/**
 * Create a compact, deterministic preview of a retrieved source passage.
 *
 * The preview keeps up to two complete sentences. If the passage contains
 * unusually long sentences, it falls back to a word-boundary truncation.
 * The full, unmodified passage remains available in the expandable source card.
 */
function getCitationExcerpt(text: string, maxLength = 360): string {
  const normalized = text.replace(/\s+/g, " ").trim();

  if (normalized.length <= maxLength) {
    return normalized;
  }

  // Split at likely sentence boundaries while avoiding most decimal splits.
  const sentences = normalized.match(/[^.!?]+[.!?]+(?:["')\]]*)|[^.!?]+$/g);

  if (sentences?.length) {
    const preview = sentences.slice(0, 2).join(" ").trim();

    if (preview.length <= maxLength) {
      return preview;
    }
  }

  // Avoid cutting a word in half when a single sentence is too long.
  const shortened = normalized.slice(0, maxLength);
  const lastSpace = shortened.lastIndexOf(" ");

  return `${shortened.slice(0, lastSpace > 0 ? lastSpace : maxLength).trim()}…`;
}

/**
 * Normalize citation markers before Markdown rendering.
 *
 * Converts legacy [E1] and numeric [1] markers into visible [1] references.
 * Matching references link to their corresponding source cards.
 */
function formatAnswerCitations(
  markdown: string,
  sourceCitations: Citation[],
  sourcePrefix: string,
): string {
  const availableSources = new Set(
    sourceCitations.map(({ evidence_id }) =>
      evidence_id.replace(/^E/i, ""),
    ),
  );

  return markdown.replace(/\[(?:E)?(\d+)\]/gi, (_match, rawId: string) => {
    const number = String(Number(rawId));

    // Link only citations backed by this response's own source cards.
    return availableSources.has(number)
      ? `[[${number}]](#${sourcePrefix}-${number})`
      : `[${number}]`;
  });
}

/**
 * Generate unique IDs for persisted conversation messages.
 *
 * The application runs in a browser, where crypto.randomUUID() is available
 * in secure contexts such as localhost and HTTPS.
 */
function generateMessageId(): string {
  return crypto.randomUUID();
}

/**
 * Render one assistant response together with its own research sources.
 *
 * Each response gets a unique source ID prefix because evidence numbers
 * can repeat across different questions in the same conversation.
 */
function AssistantResponse({
  message,
}: {
  message: AssistantMessage;
}) {
  const sourcePrefix = `source-${message.id}`;

  return (
    <section className="answer-container" aria-label="Research response">
      <div className="message-timestamp">
        {new Date(message.createdAt).toLocaleString()}
      </div>

      <div className="answer-layout">
        <main className="answer-main">
          <div className="answer-card">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex]}
            >
              {formatAnswerCitations(
                message.content,
                message.citations,
                sourcePrefix,
              )}
            </ReactMarkdown>
          </div>
        </main>

        <aside className="citations-section" aria-label="Sources">
          <div className="sources-heading">
            <h3>Sources</h3>
            <span>{message.citations.length}</span>
          </div>

          {message.citations.length === 0 ? (
            <p className="sources-empty">
              No source passages were returned.
            </p>
          ) : (
            <div className="citations-list">
              {message.citations.map((citation) => {
                const evidenceNumber = citation.evidence_id.replace(/^E/i, "");

                return (
                  <article
                    id={`${sourcePrefix}-${evidenceNumber}`}
                    className="citation-card"
                    key={citation.evidence_id}
                  >
                    <div className="citation-title-row">
                      <strong className="citation-number">
                        [{evidenceNumber}]
                      </strong>
                      <h4>{citation.title}</h4>
                    </div>

                    <div className="citation-metadata">
                      <span>arXiv: {citation.arxiv_id}</span>
                      <span
                        className="citation-metadata-separator"
                        aria-hidden="true"
                      >
                        ·
                      </span>
                      <span>Page {citation.page_number}</span>
                    </div>

                    <a
                      className="citation-paper-link"
                      href={`https://arxiv.org/abs/${encodeURIComponent(citation.arxiv_id)}`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Open paper <span aria-hidden="true">↗</span>
                    </a>

                    <p className="citation-text">
                      {getCitationExcerpt(citation.text)}
                    </p>

                    <details className="citation-original">
                      <summary>Show original passage</summary>
                      <p>{citation.text}</p>
                    </details>
                  </article>
                );
              })}
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}

function App() {
  const [currentView, setCurrentView] = useState<AppView>("chat");

  // Centralized conversation state and browser persistence.
  const {
    conversations,
    activeConversationId,
    activeConversation,
    storageError,
    createConversation,
    selectConversation,
    addMessage,
    updatePaperScope,
  } = useConversations();

  /**
   * Restore the paper context from the active conversation.
   *
   * Only the paper ID and title are required by the current chat interface.
   * The conversation stores these values as a metadata snapshot.
   */
  const selectedPaper =
    activeConversation?.arxivId
      ? {
          arxiv_id: activeConversation.arxivId,
          title: activeConversation.paperTitle ?? "Selected paper",
        }
      : null;

  // Stores the question currently being entered in the editable prompt.
  const [question, setQuestion] = useState("");
  const questionEditorRef = useRef<HTMLDivElement>(null);

  /**
   * Synchronize the editable DOM surface when the question is reset
   * programmatically, such as when starting a new conversation.
   */
  useEffect(() => {
    const editor = questionEditorRef.current;

    if (editor && editor.innerText !== question) {
      editor.innerText = question;
    }
  }, [question]);

  // Tracks whether a request is currently being processed.
  const [isLoading, setIsLoading] = useState(false);

  // Stores a user-friendly error message when a request fails.
  const [error, setError] = useState("");

  /**
   * Create a fresh unrestricted conversation.
   *
   * The conversation hook handles its ID, timestamps, and persistence.
   */
  function handleNewChat() {
    createConversation();

    setQuestion("");
    setError("");
    setCurrentView("chat");
  }

  // Open the Paper Library without discarding the active conversation.
  function handleShowLibrary() {
    setCurrentView("library");
  }

  // Open conversation history without discarding the active conversation.
  function handleShowConversations() {
    setCurrentView("conversations");
  }

  /**
   * Restore a saved conversation and return to the chat workspace.
   * The conversation hook resolves its messages and paper context
   * from the selected conversation.
   */
  function handleSelectConversation(conversationId: string) {
    selectConversation(conversationId);
    setQuestion("");
    setError("");
    setCurrentView("chat");
  }

  /**
   * Start a new conversation scoped to the selected research paper.
   *
   * The selected paper is stored in the new conversation so its context
   * can be restored when conversation history navigation is implemented.
   */
  function handleStartResearch(paper: Paper) {
    createConversation(paper.arxiv_id, paper.title);

    setQuestion("");
    setError("");
    setCurrentView("chat");
  }

  /**
   * Remove the paper filter from the active conversation.
   *
   * This does not delete previous messages or citations.
   */
  function handleClearPaperScope() {
    if (!activeConversationId) {
      return;
    }

    updatePaperScope(activeConversationId, null, null);
    setError("");
  }

  /**
   * Submit a research question to the FastAPI backend.
   *
   * Both the user question and assistant response are persisted as messages.
   * Previous messages are not yet included in the backend request; multi-turn
   * RAG context requires a separate backend implementation.
   */
  async function handleAskQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedQuestion = question.trim();

    // Do not submit empty questions or overlapping requests.
    if (!trimmedQuestion || isLoading) {
      return;
    }

    /**
     * Reuse the active conversation or create one when the user submits
     * their first question without explicitly starting a new chat.
     */
    const conversation =
      activeConversation ??
      createConversation(
        selectedPaper?.arxiv_id ?? null,
        selectedPaper?.title ?? null,
      );

    // Persist the user's question before starting backend inference.
    const userMessage: UserMessage = {
      id: generateMessageId(),
      role: "user",
      content: trimmedQuestion,
      createdAt: new Date().toISOString(),
    };

    addMessage(conversation.id, userMessage);

    setQuestion("");
    setIsLoading(true);
    setError("");

    try {
      // Send the current question and its optional paper scope.
      // Conversation history is intentionally not sent to the backend yet.
      const result = await askQuestion({
        question: trimmedQuestion,
        arxiv_id: conversation.arxivId,
      });

      // Keep the answer and its evidence together for reliable restoration.
      const assistantMessage: AssistantMessage = {
        id: generateMessageId(),
        role: "assistant",
        content: result.answer,
        citations: result.citations,
        contextTokenCount: result.context_token_count,
        createdAt: new Date().toISOString(),
      };

      addMessage(conversation.id, assistantMessage);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while processing your question.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="app-layout">
      <Sidebar
        onNewChat={handleNewChat}
        onShowLibrary={handleShowLibrary}
        onShowConversations={handleShowConversations}
        activeView={currentView}
      />

      <main className="main-content">
        {/* Persistent product identity and application description. */}
        <header className="main-header">
          <div className="brand-line">
            <h1 className="brand-name">SciLitRAG</h1>

            <span className="brand-dot" aria-hidden="true" />

            <span className="brand-role">Research Assistant</span>
          </div>

          <p className="brand-description">
            Explore scientific literature with AI-powered research.
          </p>
        </header>

        {/* Inform the user if browser storage could not persist changes. */}
        {storageError && (
          <div className="error-message" role="alert">
            <strong>Conversation storage unavailable</strong>
            <p>{storageError}</p>
          </div>
        )}

        {currentView === "library" ? (
          <PaperLibrary onStartResearch={handleStartResearch} />
        ) : currentView === "conversations" ? (
          <ConversationList
            conversations={conversations}
            onSelectConversation={handleSelectConversation}
          />
        ) : (
          <section className="chat-workspace">
            {/* Show the welcome state only for a genuinely empty conversation. */}
            {(!activeConversation || activeConversation.messages.length === 0) &&
              !isLoading &&
              !error && (
                <div className="welcome-section">
                  <h2>What would you like to research?</h2>
                  <p>
                    {selectedPaper
                      ? "Ask questions about the selected paper. Answers will use evidence retrieved from that paper."
                      : "Ask questions across your research papers and get evidence-backed answers with citations."}
                  </p>
                </div>
              )}

            {/* Render every saved message in chronological order. */}
            <div className="message-transcript">
              {activeConversation?.messages.map((message) =>
                message.role === "user" ? (
                  <article className="user-message" key={message.id}>
                    <div className="message-timestamp">
                      {new Date(message.createdAt).toLocaleString()}
                    </div>
                    <div className="user-message-content">{message.content}</div>
                  </article>
                ) : (
                  <AssistantResponse message={message} key={message.id} />
                ),
              )}
            </div>

            {isLoading && (
              <div className="loading-state" role="status">
                <p>Researching your question...</p>
              </div>
            )}

            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}

            <form className="question-form" onSubmit={handleAskQuestion}>
              {/*
               * Keep the selected-paper context inside the sticky composer.
               * This makes the context tab move with the question input.
               */}

              {selectedPaper && (
                <div className="research-scope">
                  {/* First line: context label and circular close control. */}
                  <div className="research-scope-heading">
                    <div className="research-scope-label-group">
                      <span className="research-scope-icon" aria-hidden="true">
                        <BookOpen size={15} strokeWidth={2} />
                      </span>

                      <span className="research-scope-label">
                        PAPER CONTEXT
                      </span>
                    </div>

                    <button
                      type="button"
                      className="research-scope-close"
                      onClick={handleClearPaperScope}
                      disabled={isLoading}
                      aria-label="Remove selected paper filter"
                      title="Remove paper filter"
                    >
                      <X size={15} strokeWidth={2.2} />
                    </button>
                  </div>

                  {/* Second line: title and arXiv ID always stay on one line. */}
                  <div
                    className="research-scope-details"
                    title={`${selectedPaper.title} · arXiv: ${selectedPaper.arxiv_id}`}
                  >
                    <strong className="research-scope-title">
                      {selectedPaper.title}
                    </strong>

                    <span className="research-scope-id">
                      <span className="research-scope-separator">·</span>
                      arXiv: {selectedPaper.arxiv_id}
                    </span>
                  </div>
                </div>
              )}

              <div
                ref={questionEditorRef}
                className="question-editor"
                contentEditable={!isLoading}
                suppressContentEditableWarning
                role="textbox"
                aria-label="Research question"
                aria-multiline="true"
                aria-disabled={isLoading}
                data-placeholder={
                  selectedPaper
                    ? "Ask a question about this paper..."
                    : "Ask a question about scientific research..."
                }
                onInput={(event) =>
                  setQuestion(event.currentTarget.innerText)
                }
                onKeyDown={(event) => {
                  // Enter submits; Shift+Enter inserts a line break.
                  // Do not submit while an IME composition is in progress.
                  if (
                    event.key === "Enter" &&
                    !event.shiftKey &&
                    !event.nativeEvent.isComposing
                  ) {
                    event.preventDefault();
                    event.currentTarget.closest("form")?.requestSubmit();
                  }
                }}
              />

              <div className="question-form-footer">
                <span>
                  {selectedPaper
                    ? "Research is restricted to the selected paper."
                    : "Answers are grounded in the scientific literature indexed by SciLitRAG."}
                </span>

                <button
                  type="submit"
                  disabled={!question.trim() || isLoading}
                >
                  <span>{isLoading ? "Researching..." : "Ask question"}</span>
                  <ArrowRight size={14} strokeWidth={2} />
                </button>
              </div>
            </form>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
