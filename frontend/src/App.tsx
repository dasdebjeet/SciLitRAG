
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { BookOpen, X, ArrowRight } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";

// KaTeX styles are required to display formatted mathematical expressions.
import "katex/dist/katex.min.css";

import Sidebar from "./components/Sidebar";
import PaperLibrary from "./components/PaperLibrary";

import { askQuestion } from "./services/api";
import type { Citation, Paper } from "./types/api";

import "./App.css";

type AppView = "chat" | "library";

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
): string {
  const availableSources = new Set(
    sourceCitations.map(({ evidence_id }) =>
      evidence_id.replace(/^E/i, ""),
    ),
  );

  return markdown.replace(/\[(?:E)?(\d+)\]/gi, (_match, rawId: string) => {
    const number = String(Number(rawId));

    // Link citations only when the corresponding source card exists.
    return availableSources.has(number)
      ? `[[${number}]](#source-${number})`
      : `[${number}]`;
  });
}

function App() {
  const [currentView, setCurrentView] = useState<AppView>("chat");

  // Stores the paper selected for a paper-scoped research session.
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);

  // Stores the question currently being entered in the editable prompt.
  const [question, setQuestion] = useState("");
  const questionEditorRef = useRef<HTMLDivElement>(null);

  // Synchronize the editable DOM surface when the question is reset
  // programmatically, such as when starting a new chat.
  useEffect(() => {
    const editor = questionEditorRef.current;

    if (editor && editor.innerText !== question) {
      editor.innerText = question;
    }
  }, [question]);

  // Stores the answer returned by the RAG backend.
  const [answer, setAnswer] = useState("");

  // Stores the evidence passages supporting the answer.
  const [citations, setCitations] = useState<Citation[]>([]);

  // Tracks whether a request is currently being processed.
  const [isLoading, setIsLoading] = useState(false);

  // Stores a user-friendly error message when a request fails.
  const [error, setError] = useState("");

  // Reset the current session and return to unrestricted research.
  function handleNewChat() {
    setQuestion("");
    setAnswer("");
    setCitations([]);
    setError("");
    setSelectedPaper(null);
    setCurrentView("chat");
  }

  // Open the Paper Library without discarding the current chat state.
  function handleShowLibrary() {
    setCurrentView("library");
  }

  // Start a new research session scoped to the selected paper.
  function handleStartResearch(paper: Paper) {
    setSelectedPaper(paper);
    setQuestion("");
    setAnswer("");
    setCitations([]);
    setError("");
    setCurrentView("chat");
  }

  // Remove the paper filter and return to unrestricted collection search.
  function handleClearPaperScope() {
    setSelectedPaper(null);
    setAnswer("");
    setCitations([]);
    setError("");
  }

  // Submit a research question to the FastAPI backend.
  async function handleAskQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedQuestion = question.trim();

    // Do not submit empty questions or overlapping requests.
    if (!trimmedQuestion || isLoading) {
      return;
    }

    setIsLoading(true);
    setError("");
    setAnswer("");
    setCitations([]);

    try {
      // Pass arxiv_id when a paper is selected.
      // Without it, the backend searches the complete paper collection.
      const result = await askQuestion({
        question: trimmedQuestion,
        arxiv_id: selectedPaper?.arxiv_id ?? null,
      });

      setAnswer(result.answer);
      setCitations(result.citations);
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

        {currentView === "library" ? (
          <PaperLibrary onStartResearch={handleStartResearch} />
        ) : (
          <section className="chat-workspace">
            {!answer && !isLoading && !error && (
              <div className="welcome-section">
                <h2>What would you like to research?</h2>

                <p>
                  {selectedPaper
                    ? "Ask questions about the selected paper. Answers will use evidence retrieved from that paper."
                    : "Ask questions across your research papers and get evidence-backed answers with citations."}
                </p>
              </div>
            )}

            {isLoading && (
              <div className="status-message" role="status">
                <p>Searching papers and generating your answer...</p>

                <p className="status-detail">
                  Retrieval, reranking, and local LLM inference may take a while.
                </p>
              </div>
            )}

            {error && (
              <div className="error-message" role="alert">
                <strong>Unable to answer your question</strong>
                <p>{error}</p>
              </div>
            )}

            {answer && (
              <section
                className="answer-container"
                aria-label="Research response"
              >
                <h2 className="answer-section-title">Research response</h2>

                <div className="answer-layout">
                  <main className="answer-main">
                    <div className="answer-card">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                      >
                        {formatAnswerCitations(answer, citations)}
                      </ReactMarkdown>
                    </div>
                  </main>

                  <aside className="citations-section" aria-label="Sources">
                    <div className="sources-heading">
                      <h3>Sources</h3>
                      <span>{citations.length}</span>
                    </div>

                    {citations.length === 0 ? (
                      <p className="sources-empty">
                        No source passages were returned.
                      </p>
                    ) : (
                      <div className="citations-list">
                        {citations.map((citation) => (
                          <article
                            id={`source-${citation.evidence_id.replace(/^E/i, "")}`}
                            className="citation-card"
                            key={citation.evidence_id}
                          >
                            <div className="citation-title-row">
                              <strong className="citation-number">
                                [{citation.evidence_id.replace(/^E/i, "")}]
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
                        ))}
                      </div>
                    )}
                  </aside>
                </div>
              </section>
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
                  <ArrowRight  size={14} strokeWidth={2} />
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
