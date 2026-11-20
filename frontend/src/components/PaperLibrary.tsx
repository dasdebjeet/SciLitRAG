
/*
 * PaperLibrary
 *
 * Displays the papers available in the SciLitRAG metadata catalogue.
 * Users can browse papers in Research mode or select up to three papers
 * in Compare mode for side-by-side research analysis.
 *
 * Research and comparison selection are deliberately separate:
 * - Research mode preserves the original single-paper workflow.
 * - Compare mode uses independent multi-paper selection.
 */

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";

import { Check, ChevronDown, X } from "lucide-react";

import { formatCategory, formatDate } from "../utils/formatters";
import { getPapers } from "../services/api";
import type { Paper } from "../types/api";

import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

import "./PaperLibrary.css";

interface PaperLibraryProps {
  onStartResearch: (paper: Paper) => void;
  onComparePapers?: (papers: Paper[]) => void;
}

type LibraryMode = "research" | "compare";

const MAX_COMPARISON_PAPERS = 3;

function PaperLibrary({
  onStartResearch,
  onComparePapers,
}: PaperLibraryProps) {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);

  // Comparison selection is independent of the active research paper.
  const [selectedPaperIds, setSelectedPaperIds] = useState<string[]>([]);
  const [libraryMode, setLibraryMode] = useState<LibraryMode>("research");
  const [selectionLimitReached, setSelectionLimitReached] = useState(false);

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("all");

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  const [isCategoryMenuOpen, setIsCategoryMenuOpen] = useState(false);
  const [activeCategoryIndex, setActiveCategoryIndex] = useState(0);

  const categoryFilterRef = useRef<HTMLDivElement>(null);

  // Fetch the catalogue once when the library view is mounted.
  useEffect(() => {
    let isMounted = true;

    async function loadPapers() {
      try {
        const result = await getPapers();

        if (isMounted) {
          setPapers(result);
          setSelectedPaper(result[0] ?? null);
        }
      } catch (err) {
        if (isMounted) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load the paper library.",
          );
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }

    void loadPapers();

    return () => {
      isMounted = false;
    };
  }, []);

  // Build category options from the available catalogue.
  const categories = useMemo(() => {
    const uniqueCategories = new Set<string>();

    papers.forEach((paper) => {
      paper.categories.forEach((category) => {
        uniqueCategories.add(category);
      });
    });

    return Array.from(uniqueCategories).sort((a, b) =>
      a.localeCompare(b),
    );
  }, [papers]);

  const categoryOptions = useMemo(
    () => [
      { value: "all", label: "All categories" },
      ...categories.map((category) => ({
        value: category,
        label: formatCategory(category),
      })),
    ],
    [categories],
  );

  const selectedCategoryLabel =
    categoryOptions.find((option) => option.value === selectedCategory)
      ?.label ?? "All categories";

  // Resolve comparison selections in the order in which they were selected.
  const selectedPapers = useMemo(
    () =>
      selectedPaperIds
        .map((id) => papers.find((paper) => paper.arxiv_id === id))
        .filter((paper): paper is Paper => paper !== undefined),
    [papers, selectedPaperIds],
  );

  const openCategoryMenu = () => {
    const selectedIndex = categoryOptions.findIndex(
      (option) => option.value === selectedCategory,
    );

    setActiveCategoryIndex(Math.max(0, selectedIndex));
    setIsCategoryMenuOpen(true);
  };

  const handleCategorySelect = (category: string) => {
    setSelectedCategory(category);
    setIsCategoryMenuOpen(false);
  };

  // Support keyboard navigation through category options.
  const handleCategoryKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
  ) => {
    const lastIndex = categoryOptions.length - 1;

    if (event.key === "Escape") {
      event.preventDefault();
      setIsCategoryMenuOpen(false);
      return;
    }

    if (!isCategoryMenuOpen) {
      if (
        event.key === "ArrowDown" ||
        event.key === "ArrowUp" ||
        event.key === "Enter" ||
        event.key === " "
      ) {
        event.preventDefault();
        openCategoryMenu();
      }

      return;
    }

    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        setActiveCategoryIndex((index) => Math.min(index + 1, lastIndex));
        break;

      case "ArrowUp":
        event.preventDefault();
        setActiveCategoryIndex((index) => Math.max(index - 1, 0));
        break;

      case "Home":
        event.preventDefault();
        setActiveCategoryIndex(0);
        break;

      case "End":
        event.preventDefault();
        setActiveCategoryIndex(lastIndex);
        break;

      case "Enter":
      case " ":
        event.preventDefault();
        handleCategorySelect(categoryOptions[activeCategoryIndex].value);
        break;
    }
  };

  // Close the category dropdown when the user clicks outside it.
  useEffect(() => {
    if (!isCategoryMenuOpen) {
      return;
    }

    const handleOutsideClick = (event: PointerEvent) => {
      if (
        categoryFilterRef.current &&
        event.target instanceof Node &&
        !categoryFilterRef.current.contains(event.target)
      ) {
        setIsCategoryMenuOpen(false);
      }
    };

    document.addEventListener("pointerdown", handleOutsideClick);

    return () => {
      document.removeEventListener("pointerdown", handleOutsideClick);
    };
  }, [isCategoryMenuOpen]);

  // Apply text search and category filtering to the catalogue.
  const filteredPapers = useMemo(() => {
    const normalizedQuery = searchQuery.trim().toLowerCase();

    return papers.filter((paper) => {
      const matchesCategory =
        selectedCategory === "all" ||
        paper.categories.includes(selectedCategory);

      const searchableText = [
        paper.title,
        paper.arxiv_id,
        paper.abstract,
        paper.primary_category,
        ...paper.authors,
        ...paper.categories,
        ...paper.categories.map(formatCategory),
        formatCategory(paper.primary_category),
      ]
        .join(" ")
        .toLowerCase();

      const matchesSearch =
        !normalizedQuery || searchableText.includes(normalizedQuery);

      return matchesCategory && matchesSearch;
    });
  }, [papers, searchQuery, selectedCategory]);

  /*
   * Keep the original details-panel behaviour in Research mode:
   * display the active paper if it is still visible, otherwise the first
   * result matching the current filters.
   */
  const displayedPaper =
    filteredPapers.find(
      (paper) => paper.arxiv_id === selectedPaper?.arxiv_id,
    ) ??
    filteredPapers[0] ??
    null;

  // Switch modes without allowing comparison selections to leak into research.
  const handleModeChange = (mode: LibraryMode) => {
    if (mode === libraryMode) {
      return;
    }

    setLibraryMode(mode);
    setSelectedPaperIds([]);
    setSelectionLimitReached(false);
  };

  /*
   * Paper clicks have different meanings depending on the active mode.
   * Research: update the single active paper.
   * Compare: add/remove the paper from the comparison selection.
   */
  const handlePaperSelection = (paper: Paper) => {
    if (libraryMode === "research") {
      setSelectedPaper(paper);
      return;
    }

    setSelectionLimitReached(false);

    setSelectedPaperIds((currentIds) => {
      if (currentIds.includes(paper.arxiv_id)) {
        return currentIds.filter((id) => id !== paper.arxiv_id);
      }

      if (currentIds.length >= MAX_COMPARISON_PAPERS) {
        setSelectionLimitReached(true);
        return currentIds;
      }

      return [...currentIds, paper.arxiv_id];
    });
  };

  const handleRemovePaper = (arxivId: string) => {
    setSelectionLimitReached(false);
    setSelectedPaperIds((currentIds) =>
      currentIds.filter((id) => id !== arxivId),
    );
  };

  const handleClearSelection = () => {
    setSelectedPaperIds([]);
    setSelectionLimitReached(false);
  };

  const handleComparePapers = () => {
    if (selectedPapers.length < 2 || !onComparePapers) {
      return;
    }

    onComparePapers(selectedPapers);
  };

  if (isLoading) {
    return (
      <section className="library-workspace">
        <div className="library-status" role="status">
          Loading your research papers...
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="library-workspace">
        <div className="library-error" role="alert">
          <strong>Unable to load the paper library</strong>
          <p>{error}</p>
          <button
            type="button"
            onClick={() => window.location.reload()}
          >
            Reload application
          </button>
        </div>
      </section>
    );
  }

  return (
    <section className="library-workspace">
      <header className="library-header">
        <div>
          <h2>Paper Library</h2>
          <p>
            Browse and explore the scientific papers in your collection.
          </p>
        </div>

        {/* Reuse the existing Settings theme-toggle design. */}
        <div
          className="settings-theme-toggle library-mode-toggle"
          role="group"
          aria-label="Paper library mode"
        >
          <button
            type="button"
            className={`settings-theme-option ${
              libraryMode === "research" ? "active" : ""
            }`}
            onClick={() => handleModeChange("research")}
            aria-pressed={libraryMode === "research"}
          >
            <span>Research</span>
          </button>

          <button
            type="button"
            className={`settings-theme-option ${
              libraryMode === "compare" ? "active" : ""
            }`}
            onClick={() => handleModeChange("compare")}
            aria-pressed={libraryMode === "compare"}
          >
            <span>Compare</span>
          </button>
        </div>
      </header>

      <div className="library-toolbar">
        <input
          type="search"
          value={searchQuery}
          onChange={(event) => setSearchQuery(event.target.value)}
          placeholder="Search titles, authors, abstracts, or arXiv IDs..."
          aria-label="Search papers"
        />

        <div
          className={`category-filter ${
            isCategoryMenuOpen ? "is-open" : ""
          }`}
          ref={categoryFilterRef}
        >
          <button
            type="button"
            className="category-filter-trigger"
            role="combobox"
            aria-label="Filter papers by category"
            aria-haspopup="listbox"
            aria-expanded={isCategoryMenuOpen}
            aria-controls="paper-category-options"
            aria-activedescendant={
              isCategoryMenuOpen
                ? `category-option-${activeCategoryIndex}`
                : undefined
            }
            onClick={() => {
              if (isCategoryMenuOpen) {
                setIsCategoryMenuOpen(false);
              } else {
                openCategoryMenu();
              }
            }}
            onKeyDown={handleCategoryKeyDown}
          >
            <span className="category-filter-value">
              {selectedCategoryLabel}
            </span>

            <ChevronDown
              size={16}
              strokeWidth={1.8}
              className="category-filter-chevron"
              aria-hidden="true"
            />
          </button>

          {isCategoryMenuOpen && (
            <div className="category-dropdown">
              <div className="category-dropdown-heading">
                FILTER BY CATEGORY
              </div>

              <div
                id="paper-category-options"
                className="category-dropdown-options"
                role="listbox"
                aria-label="Paper categories"
              >
                {categoryOptions.map((option, index) => {
                  const isSelected =
                    selectedCategory === option.value;

                  return (
                    <button
                      key={option.value}
                      id={`category-option-${index}`}
                      type="button"
                      role="option"
                      aria-selected={isSelected}
                      className={`category-option ${
                        isSelected ? "selected" : ""
                      } ${
                        activeCategoryIndex === index ? "active" : ""
                      }`}
                      onMouseEnter={() => setActiveCategoryIndex(index)}
                      onClick={() => handleCategorySelect(option.value)}
                    >
                      <span>{option.label}</span>

                      {isSelected && (
                        <Check
                          size={15}
                          strokeWidth={2.2}
                          aria-hidden="true"
                        />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="library-content">
        {/* Left column: searchable paper catalogue. */}
        <div className="paper-results">
          <div className="paper-results-heading">
            <strong>Research papers</strong>
            <span>{filteredPapers.length} results</span>
          </div>

          {filteredPapers.length === 0 ? (
            <div className="library-empty">
              No papers match your search.
            </div>
          ) : (
            <div className="paper-list">
              {filteredPapers.map((paper) => {
                const isComparisonSelected =
                  selectedPaperIds.includes(paper.arxiv_id);

                const isHighlighted =
                  libraryMode === "compare"
                    ? isComparisonSelected
                    : displayedPaper?.arxiv_id === paper.arxiv_id;

                const isSelectionLimitReached =
                  libraryMode === "compare" &&
                  selectedPaperIds.length >= MAX_COMPARISON_PAPERS &&
                  !isComparisonSelected;

                return (
                  <button
                    type="button"
                    key={paper.arxiv_id}
                    className={`paper-list-item ${
                      isHighlighted ? "selected" : ""
                    }`}
                    onClick={() => handlePaperSelection(paper)}
                    aria-pressed={
                      libraryMode === "compare"
                        ? isComparisonSelected
                        : displayedPaper?.arxiv_id === paper.arxiv_id
                    }
                    title={
                      libraryMode === "research"
                        ? "View paper details"
                        : isSelectionLimitReached
                          ? "You can compare up to three papers."
                          : isComparisonSelected
                            ? "Remove from comparison"
                            : "Select for comparison"
                    }
                  >
                    <span className="paper-list-title">
                      <ReactMarkdown
                        remarkPlugins={[remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                        components={{
                          p: ({ children }) => <>{children}</>,
                        }}
                      >
                        {paper.title}
                      </ReactMarkdown>
                    </span>

                    <span className="paper-list-meta">
                      {paper.arxiv_id} · {formatDate(paper.published)}
                    </span>

                    <span className="paper-list-authors">
                      {paper.authors.join(", ")}
                    </span>

                    {libraryMode === "compare" && isComparisonSelected && (
                      <span className="paper-selection-indicator">
                        <Check size={13} strokeWidth={2.4} aria-hidden="true" />
                        Selected
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Right column: comparison panel or normal paper details. */}
        <aside className="paper-details">
          {libraryMode === "compare" ? (
            <div className="comparison-selection">
              <div className="comparison-selection-header">
                <div>
                  <span className="paper-details-label">
                    COMPARISON
                  </span>

                  <h3>Selected papers</h3>

                  <p>
                    Choose two or three papers to compare their research,
                    methodology, and findings.
                  </p>
                </div>

                <span className="comparison-selection-count">
                  {selectedPapers.length}/{MAX_COMPARISON_PAPERS}
                </span>
              </div>

              {selectedPapers.length > 0 ? (
                <div className="comparison-selected-list">
                  {selectedPapers.map((paper, index) => (
                    <article
                      className="comparison-selected-paper"
                      key={paper.arxiv_id}
                    >
                      <div className="comparison-paper-heading">
                        <span className="comparison-paper-index">
                          {String(index + 1).padStart(2, "0")}
                        </span>

                        <h4>
                          <ReactMarkdown
                            remarkPlugins={[remarkMath]}
                            rehypePlugins={[rehypeKatex]}
                            components={{
                              p: ({ children }) => <>{children}</>,
                            }}
                          >
                            {paper.title}
                          </ReactMarkdown>
                        </h4>

                        <button
                          type="button"
                          className="comparison-remove-button"
                          onClick={() => handleRemovePaper(paper.arxiv_id)}
                          aria-label={`Remove ${paper.title} from comparison`}
                          title="Remove paper"
                        >
                          <X size={15} strokeWidth={1.9} />
                        </button>
                      </div>

                      <div className="comparison-paper-metadata">
                        <a
                          href={paper.pdf_url}
                          target="_blank"
                          rel="noreferrer"
                          className="comparison-paper-arxiv-link"
                          aria-label={`Open PDF for arXiv paper ${paper.arxiv_id}`}
                        >
                          arXiv: {paper.arxiv_id}
                        </a>

                        <span aria-hidden="true">·</span>

                        <span>
                          Published {formatDate(paper.published)}
                        </span>

                        <span aria-hidden="true">·</span>

                        <span>
                          {formatCategory(paper.primary_category)}
                        </span>
                      </div>

                      <div className="comparison-paper-authors">
                        <span>Authors</span>
                        <p>{paper.authors.join(", ")}</p>
                      </div>
                    </article>
                  ))}
                </div>
              ) : (
                <div className="comparison-empty-state">
                  <div className="comparison-empty-icon">
                    <Check size={20} strokeWidth={1.7} />
                  </div>

                  <h4>No papers selected</h4>

                  <p>
                    Select papers from the list to start building your
                    comparison. You can select up to three papers.
                  </p>
                </div>
              )}

              <div className="comparison-selection-footer">
                {selectionLimitReached && (
                  <p className="comparison-limit-message" role="status">
                    You can compare a maximum of three papers.
                  </p>
                )}

                {selectedPapers.length > 0 && (
                  <button
                    type="button"
                    className="comparison-clear-button"
                    onClick={handleClearSelection}
                  >
                    Clear selection
                  </button>
                )}

                <button
                  type="button"
                  className="start-research-button comparison-action-button"
                  onClick={handleComparePapers}
                  disabled={
                    selectedPapers.length < 2 || !onComparePapers
                  }
                  title={
                    selectedPapers.length < 2
                      ? "Select at least two papers to compare."
                      : !onComparePapers
                        ? "Comparison action has not been connected yet."
                        : "Compare selected papers"
                  }
                >
                  Compare selected papers
                  {selectedPapers.length >= 2 && (
                    <span>({selectedPapers.length})</span>
                  )}
                </button>
              </div>
            </div>
          ) : displayedPaper ? (
            <>
              <div className="paper-details-heading">
                <span className="paper-details-label">
                  PAPER DETAILS
                </span>

                <h3 className="paper-title">
                  <ReactMarkdown
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                    components={{
                      p: ({ children }) => <>{children}</>,
                    }}
                  >
                    {displayedPaper.title}
                  </ReactMarkdown>
                </h3>
              </div>

              <div className="paper-metadata">
                <div>
                  <span>arXiv ID</span>
                  <strong>{displayedPaper.arxiv_id}</strong>
                </div>

                <div>
                  <span>Published</span>
                  <strong>{formatDate(displayedPaper.published)}</strong>
                </div>

                <div>
                  <span>Primary category</span>
                  <strong>
                    {formatCategory(displayedPaper.primary_category)}
                  </strong>
                </div>

                <div>
                  <span>Authors</span>
                  <strong>{displayedPaper.authors.join(", ")}</strong>
                </div>

                <a
                  href={displayedPaper.pdf_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  View paper on arXiv
                </a>
              </div>

              <div className="paper-abstract">
                <h4>Abstract</h4>

                <div>
                  <ReactMarkdown
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                  >
                    {displayedPaper.abstract}
                  </ReactMarkdown>
                </div>
              </div>

              <div className="paper-category-list">
                {displayedPaper.categories.map((category) => (
                  <span key={category}>
                    {formatCategory(category)}
                  </span>
                ))}
              </div>

              <div className="paper-details-actions">
                <button
                  type="button"
                  className="start-research-button"
                  onClick={() => onStartResearch(displayedPaper)}
                >
                  Start research with this paper
                </button>
              </div>
            </>
          ) : (
            <div className="library-empty">
              Select a paper to view its details.
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}

export default PaperLibrary;
