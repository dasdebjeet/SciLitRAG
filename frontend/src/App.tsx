
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { BookOpen, X, ArrowRight, ArrowDown } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";

// KaTeX styles are required to display formatted mathematical expressions.
import "katex/dist/katex.min.css";

import ConversationList from "./components/conversations/ConversationList";
import Sidebar from "./components/Sidebar";
import PaperLibrary from "./components/PaperLibrary";

import SettingsPanel, {
  type ThemePreference,
} from "./components/SettingsPanel";

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
        {/* {new Date(message.createdAt).toLocaleString()} */}
          <time dateTime={message.createdAt}>
            {new Date(message.createdAt).toLocaleString()}
          </time>

          {message.generationDurationSeconds != null && (
            <>
              <span
                className="message-metadata-separator"
                aria-hidden="true"
              />
              <span>
                {message.generationDurationSeconds.toFixed(2)}s
              </span>
            </>
          )}

          {message.tokensPerSecond != null && (
            <>
              <span
                className="message-metadata-separator"
                aria-hidden="true"
              />
              <span>
                {message.tokensPerSecond.toFixed(1)} tokens/s
              </span>
            </>
          )}
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
    renameConversation,
    updatePaperScope,
    deleteConversation,
    clearAllConversations,
  } = useConversations();

  // Stores the question currently being entered in the editable prompt.
  const [question, setQuestion] = useState("");

  // Holds the paper scope for a new, unsaved chat draft.
  const [draftPaper, setDraftPaper] = useState<Paper | null>(null);

  // References the editable prompt, sticky composer, and message transcript.
  const questionEditorRef = useRef<HTMLDivElement>(null);
  const questionFormRef = useRef<HTMLFormElement>(null);
  const messageTranscriptRef = useRef<HTMLDivElement>(null);

  // Controls the visibility and vertical position of the floating scroll button.
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const [scrollButtonBottom, setScrollButtonBottom] = useState(180);

  // Tracks whether a request is currently being processed.
  const [isLoading, setIsLoading] = useState(false);

  // Stores a user-friendly error message when a request fails.
  const [error, setError] = useState("");

  /**
   * Restore paper scope from a saved conversation or the current unsaved draft.
   *
   * An active saved conversation takes priority. Otherwise, the paper
   * selected for the unsaved draft is used until its first message is sent.
   */
  const selectedPaper = activeConversation
    ? activeConversation.arxivId
      ? {
          arxiv_id: activeConversation.arxivId,
          title: activeConversation.paperTitle ?? "Selected paper",
        }
      : null
    : draftPaper;

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Restore the user's saved theme preference; fall back to system settings.
  const [themePreference, setThemePreference] = useState<ThemePreference>(() => {
    try {
      const savedTheme = window.localStorage.getItem("scilitrag-theme");

      if (
        savedTheme === "light" ||
        savedTheme === "dark" ||
        savedTheme === "system"
      ) {
        return savedTheme;
      }
    } catch (error) {
      console.warn("Unable to read the saved theme preference.", error);
    }

    return "system";
  });

  // Explicit themes override the OS preference. System removes the override.
  useEffect(() => {
    const root = document.documentElement;

    if (themePreference === "system") {
      root.removeAttribute("data-theme");
    } else {
      root.setAttribute("data-theme", themePreference);
    }

    try {
      window.localStorage.setItem("scilitrag-theme", themePreference);
    } catch (error) {
      console.warn("Unable to save the theme preference.", error);
    }
  }, [themePreference]);

  /**
   * Monitor document scrolling and chat layout changes.
   *
   * The application uses normal document scrolling rather than an
   * independently scrollable message container. Measuring the document
   * preserves the existing page layout and scrolling behavior.
   */
  useEffect(() => {
    // Scroll tracking is only needed while the chat view is visible.
    // Button visibility is derived from currentView in the render below.
    if (currentView !== "chat") {
      return;
    }

    function updateScrollButton() {
      const viewportBottom = window.scrollY + window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight;

      // Only show the button when the user is meaningfully away from the bottom.
      setShowScrollToBottom(documentHeight - viewportBottom > 180);

      // Keep the button just above the sticky composer.
      const composerTop =
        questionFormRef.current?.getBoundingClientRect().top;

      if (composerTop !== undefined) {
        const nextBottom = Math.max(
          100,
          window.innerHeight - composerTop + 12,
        );

        setScrollButtonBottom((previous) =>
          Math.abs(previous - nextBottom) > 1 ? nextBottom : previous,
        );
      }
    }

    // Calculate the initial visibility and position.
    updateScrollButton();

    // Track page scrolling and viewport resizing.
    window.addEventListener("scroll", updateScrollButton, { passive: true });
    window.addEventListener("resize", updateScrollButton);

    // Observe content changes that can move the composer or page bottom.
    const resizeObserver =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(updateScrollButton)
        : null;

    if (questionFormRef.current) {
      resizeObserver?.observe(questionFormRef.current);
    }

    if (messageTranscriptRef.current) {
      resizeObserver?.observe(messageTranscriptRef.current);
    }

    return () => {
      window.removeEventListener("scroll", updateScrollButton);
      window.removeEventListener("resize", updateScrollButton);
      resizeObserver?.disconnect();
    };
  }, [
    currentView,
    activeConversation?.messages.length,
    isLoading,
    selectedPaper?.arxiv_id,
  ]);

  /**
   * Scroll to the latest conversation content.
   *
   * Respect the operating system's reduced-motion accessibility preference
   * instead of forcing smooth animation for users who have disabled motion.
   */
  function handleScrollToBottom() {
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: prefersReducedMotion ? "auto" : "smooth",
    });
  }

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

  /**
   * Start a new, unsaved conversation draft.
   *
   * No conversation is created or persisted until the user submits
   * the first message.
   */
  function handleNewChat() {
    selectConversation(null);
    setDraftPaper(null);
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
   * Any unsaved draft paper context is discarded.
   */
  function handleSelectConversation(conversationId: string) {
    selectConversation(conversationId);
    setDraftPaper(null);
    setQuestion("");
    setError("");
    setCurrentView("chat");
  }

  /**
   * Start an unsaved conversation scoped to the selected research paper.
   *
   * The paper metadata remains in draft state until the first question
   * creates a persisted conversation.
   */
  function handleStartResearch(paper: Paper) {
    selectConversation(null);
    setDraftPaper(paper);

    setQuestion("");
    setError("");
    setCurrentView("chat");
  }

  /**
   * Remove the paper filter from the active conversation or unsaved draft.
   *
   * Existing messages and citations are preserved.
   */
  function handleClearPaperScope() {
    if (activeConversationId) {
      updatePaperScope(activeConversationId, null, null);
    } else {
      setDraftPaper(null);
    }

    setError("");
  }

  /**
   * Submit a research question to the FastAPI backend.
   *
   * The first submitted message creates and persists a conversation.
   * Subsequent messages reuse the active conversation and its history.
   */
  async function handleAskQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedQuestion = question.trim();

    // Do not submit empty questions or overlapping requests.
    if (!trimmedQuestion || isLoading) {
      return;
    }

    /**
     * Reuse the active conversation or create one when this is the
     * first message in a new draft.
     *
     * The selected paper metadata is copied into the persisted conversation.
     */
    const conversation =
      activeConversation ??
      createConversation(
        selectedPaper?.arxiv_id ?? null,
        selectedPaper?.title ?? null,
      );

    // The draft has now become a persisted conversation.
    setDraftPaper(null);

    // Keep recent conversation context without resending excessively long answers.
    // History is used to understand follow-up questions, not as scientific evidence.
    const history = conversation.messages.slice(-12).map((message) => ({
      role: message.role,
      content:
        message.content.length > 2000
          ? `${message.content.slice(0, 2000)}\n[Message truncated for conversation context]`
          : message.content,
    }));

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
      // Send the current question, its paper scope, and recent conversation context.
      const result = await askQuestion({
        question: trimmedQuestion,
        arxiv_id: conversation.arxivId,
        history,
      });

      // Keep the answer and its evidence together for reliable restoration.
      const assistantMessage: AssistantMessage = {
        id: generateMessageId(),
        role: "assistant",
        content: result.answer,
        citations: result.citations,
        contextTokenCount: result.context_token_count,
        createdAt: new Date().toISOString(),

        // Preserve the exact generation statistics returned by the backend.
        // Missing values remain undefined instead of displaying placeholders.
        generationDurationSeconds:
          result.generation_duration_seconds ?? undefined,

        generationTokenCount:
          result.generation_token_count ?? undefined,

        tokensPerSecond:
          result.tokens_per_second ?? undefined,
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
        onToggleSettings={() => setIsSettingsOpen((open) => !open)}
        activeView={currentView}
        settingsOpen={isSettingsOpen}
      />

      <SettingsPanel
        key={isSettingsOpen ? "settings-open" : "settings-closed"}
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        theme={themePreference}
        onThemeChange={setThemePreference}
        conversationCount={conversations.length}
        onClearConversations={clearAllConversations}
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
            onRenameConversation={renameConversation}
            onDeleteConversation={deleteConversation}
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
            <div
              ref={messageTranscriptRef}
              className="message-transcript"
            >
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

            {/* Floating control for navigating long conversations. */}
            {currentView === "chat" && showScrollToBottom && (
              <button
                type="button"
                className="scroll-to-bottom-button"
                onClick={handleScrollToBottom}
                aria-label="Scroll to latest message"
                title="Scroll to bottom"
                style={{ bottom: `${scrollButtonBottom}px` }}
              >
                <ArrowDown size={20} strokeWidth={2.2} />
              </button>
            )}

            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}

            <form
              ref={questionFormRef}
              className="question-form"
              onSubmit={handleAskQuestion}
            >
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
