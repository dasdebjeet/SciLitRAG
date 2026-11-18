
/*
 * PaperLibrary
 *
 * Displays the papers available in the SciLitRAG metadata catalogue.
 * Users can search the collection, filter by category, inspect paper
 * details, and start a research chat scoped to a selected paper.
 *
 * The category filter uses a custom dropdown so its appearance remains
 * consistent with the application's dark and light themes.
 */

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";

import { Check, ChevronDown } from "lucide-react";

import { formatCategory, formatDate } from "../utils/formatters";
import { getPapers } from "../services/api";
import type { Paper } from "../types/api";

import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

import "./PaperLibrary.css";

interface PaperLibraryProps {
  // The parent owns navigation and chat state.
  onStartResearch: (paper: Paper) => void;
}

function PaperLibrary({ onStartResearch }: PaperLibraryProps) {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("all");

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  // Track the custom category dropdown's visibility and keyboard selection.
  const [isCategoryMenuOpen, setIsCategoryMenuOpen] = useState(false);
  const [activeCategoryIndex, setActiveCategoryIndex] = useState(0);

  // Used to detect pointer interactions outside the category dropdown.
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

    // Prevent state updates if the user leaves the library while loading.
    return () => {
      isMounted = false;
    };
  }, []);

  // Build the category options from the actual catalogue.
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

  // Build dropdown options from the catalogue and include the unfiltered option.
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

  // Resolve the displayed label from the currently selected category.
  const selectedCategoryLabel =
    categoryOptions.find((option) => option.value === selectedCategory)
      ?.label ?? "All categories";

  // Open the dropdown and highlight the currently selected category.
  const openCategoryMenu = () => {
    const selectedIndex = categoryOptions.findIndex(
      (option) => option.value === selectedCategory,
    );

    setActiveCategoryIndex(Math.max(0, selectedIndex));
    setIsCategoryMenuOpen(true);
  };

  // Apply a category selection and close the dropdown.
  const handleCategorySelect = (category: string) => {
    setSelectedCategory(category);
    setIsCategoryMenuOpen(false);
  };

  // Support keyboard navigation through the available category options.
  const handleCategoryKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
  ) => {
    const lastIndex = categoryOptions.length - 1;

    if (event.key === "Escape") {
      event.preventDefault();
      setIsCategoryMenuOpen(false);
      return;
    }

    // Allow keyboard users to open the dropdown from its closed state.
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

  // Close the dropdown when the user interacts outside its boundary.
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

  // Search across the most useful bibliographic fields and apply category filtering.
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
   * Keep the details panel synchronized with the filtered results
   * without introducing an additional state update or render cycle.
   *
   * Preserve the user's selection when it remains visible. Otherwise,
   * display the first matching paper, or null when no results exist.
   */
  const displayedPaper =
    filteredPapers.find(
      (paper) => paper.arxiv_id === selectedPaper?.arxiv_id,
    ) ??
    filteredPapers[0] ??
    null;

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

        <span className="library-count">
          {papers.length} papers
        </span>
      </header>

      <div className="library-toolbar">
        <input
          type="search"
          value={searchQuery}
          onChange={(event) => setSearchQuery(event.target.value)}
          placeholder="Search titles, authors, abstracts, or arXiv IDs..."
          aria-label="Search papers"
        />

        {/* Custom category dropdown with theme-aware styling. */}
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
              {filteredPapers.map((paper) => (
                <button
                  type="button"
                  key={paper.arxiv_id}
                  className={`paper-list-item ${
                    displayedPaper?.arxiv_id === paper.arxiv_id
                      ? "selected"
                      : ""
                  }`}
                  onClick={() => setSelectedPaper(paper)}
                >
                  <span className="paper-list-title">
                    {/* Render LaTeX in titles while preserving the compact list-item layout. */}
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
                </button>
              ))}
            </div>
          )}
        </div>

        <aside className="paper-details">
          {displayedPaper ? (
            <>
              <div className="paper-details-heading">
                <span className="paper-details-label">
                  PAPER DETAILS
                </span>

                <h3 className="paper-title">
                  {/* Support LaTeX expressions embedded in paper titles. */}
                  <ReactMarkdown
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                    components={{
                      // Keep the title as a heading rather than creating nested headings.
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
              </div>

              <div className="paper-abstract">
                <h4>Abstract</h4>

                <div className="paper-abstract">
                  {/* Render abstract text as Markdown and convert LaTeX expressions into math. */}
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

                <a
                  href={displayedPaper.pdf_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  View paper on arXiv
                </a>
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
