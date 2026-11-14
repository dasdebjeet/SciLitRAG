
"""
Fetch research papers from arXiv and maintain a local PDF/metadata corpus.

Design goals:
    - Search across configurable arXiv categories.
    - Avoid duplicate downloads across repeated runs.
    - Validate downloaded PDFs before accepting them.
    - Retry transient network failures.
    - Preserve existing metadata instead of overwriting the corpus.
    - Write metadata atomically to reduce the risk of file corruption.

Usage:
    uv run python scripts/fetch_papers.py
    uv run python scripts/fetch_papers.py --max-results 120
    uv run python scripts/fetch_papers.py --max-results 150 --categories cs.LG cs.AI cs.CV cs.CL
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import arxiv
import pymupdf


# Resolve project paths relative to this script, not the shell's current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
METADATA_FILE = METADATA_DIR / "papers.json"

# Keep the initial default small. A larger corpus must be requested explicitly.
DEFAULT_CATEGORIES = ["cs.LG", "cs.AI", "cs.CV", "cs.CL"]
DEFAULT_MAX_RESULTS = 5

# A descriptive user agent helps identify this downloader if a server reports
# unusual traffic. The timeout prevents a stalled connection from hanging forever.
USER_AGENT = "SciLitRAG/1.0 (research paper ingestion)"
DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_DELAY_SECONDS = 1.0

logger = logging.getLogger("scilit_rag.fetch_papers")


def configure_logging() -> None:
    """Configure consistent console logging for the ingestion script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options so corpus size and scope are configurable."""
    parser = argparse.ArgumentParser(
        description="Download and validate research papers from arXiv."
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help=f"Maximum number of search results to process (default: {DEFAULT_MAX_RESULTS}).",
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="arXiv categories to search, e.g. cs.LG cs.AI cs.CV cs.CL.",
    )

    parser.add_argument(
        "--sort-by",
        choices=("submitted", "relevance"),
        default="submitted",
        help="Sort results by submission date or relevance.",
    )

    parser.add_argument(
        "--max-download-retries",
        type=int,
        default=3,
        help="Number of attempts for each PDF download (default: 3).",
    )

    args = parser.parse_args()

    if args.max_results < 1:
        parser.error("--max-results must be at least 1.")

    if args.max_download_retries < 1:
        parser.error("--max-download-retries must be at least 1.")

    if not args.categories:
        parser.error("At least one arXiv category must be provided.")

    return args


def configure_directories() -> None:
    """Create required directories without deleting existing corpus data."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def load_existing_metadata() -> list[dict[str, Any]]:
    """
    Load previously collected metadata.

    Returning an empty list for a missing file supports the first ingestion run.
    Malformed metadata is treated as a hard error rather than silently discarded,
    because overwriting a corrupted file could destroy valuable corpus records.
    """
    if not METADATA_FILE.exists():
        return []

    try:
        with METADATA_FILE.open("r", encoding="utf-8") as file:
            records = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"Could not safely read metadata file: {METADATA_FILE}"
        ) from exc

    if not isinstance(records, list):
        raise ValueError(
            f"Expected a JSON list in {METADATA_FILE}, got {type(records).__name__}."
        )

    return records


def save_metadata_atomically(records: list[dict[str, Any]]) -> None:
    """
    Persist metadata using a temporary file followed by an atomic replacement.

    This avoids exposing a partially written JSON file if the process is
    interrupted while saving the corpus metadata.
    """
    temporary_file = METADATA_FILE.with_suffix(".json.tmp")

    try:
        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2, ensure_ascii=False)
            file.write("\n")

        temporary_file.replace(METADATA_FILE)

    except OSError:
        # Preserve the original metadata file if the replacement fails.
        temporary_file.unlink(missing_ok=True)
        raise


def get_base_arxiv_id(arxiv_id: str) -> str:
    """
    Remove the version suffix from an arXiv identifier.

    For example, 2401.12345v2 and 2401.12345v1 share the same base ID.
    This prevents different versions of one paper from entering the corpus
    as separate documents.
    """
    return re.sub(r"v\d+$", "", arxiv_id)


def get_existing_paper_ids(records: list[dict[str, Any]]) -> set[str]:
    """
    Identify papers already represented by usable local PDF files.

    A metadata entry only counts as existing when its referenced PDF is present.
    This permits a missing PDF to be downloaded again on a later run.
    """
    existing_ids: set[str] = set()

    for record in records:
        arxiv_id = record.get("arxiv_id")
        pdf_path_value = record.get("pdf_path")

        if not arxiv_id or not pdf_path_value:
            logger.warning("Skipping metadata entry with missing ID or PDF path.")
            continue

        pdf_path = PROJECT_ROOT / Path(pdf_path_value)

        if pdf_path.is_file() and pdf_path.stat().st_size > 0:
            existing_ids.add(get_base_arxiv_id(str(arxiv_id)))

    return existing_ids


def validate_pdf(pdf_path: Path) -> tuple[bool, str]:
    """
    Check whether a downloaded file is a readable, non-empty PDF.

    PDF magic bytes provide a quick format check. PyMuPDF then verifies that
    the document can actually be opened and contains at least one page.
    Text extraction is intentionally not required here: some legitimate
    research PDFs contain scanned pages or figures with little selectable text.
    """
    if not pdf_path.is_file():
        return False, "File does not exist."

    if pdf_path.stat().st_size == 0:
        return False, "File is empty."

    try:
        with pdf_path.open("rb") as file:
            header = file.read(1024)

        if b"%PDF-" not in header:
            return False, "PDF signature not found."

        with pymupdf.open(str(pdf_path)) as document:
            if document.is_encrypted:
                return False, "Encrypted PDF is not supported."

            if document.page_count < 1:
                return False, "PDF contains no pages."

    except Exception as exc:
        return False, f"PDF could not be opened: {exc}"

    return True, "Valid PDF."


def download_pdf(
    pdf_url: str,
    destination: Path,
    max_retries: int,
) -> bool:
    """
    Download one PDF with retries and validate it before accepting it.

    Data is written to a temporary .part file first. The final destination
    is only replaced after the temporary file passes PDF validation.
    """
    temporary_path = destination.with_suffix(".pdf.part")
    request = Request(
        pdf_url,
        headers={"User-Agent": USER_AGENT},
    )

    for attempt in range(1, max_retries + 1):
        try:
            temporary_path.unlink(missing_ok=True)

            logger.info(
                "Downloading PDF (attempt %d/%d): %s",
                attempt,
                max_retries,
                pdf_url,
            )

            with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                expected_size = response.headers.get("Content-Length")
                downloaded_size = 0

                with temporary_path.open("wb") as output_file:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)

                        if not chunk:
                            break

                        output_file.write(chunk)
                        downloaded_size += len(chunk)

                # Detect a truncated response when the server provides a
                # Content-Length header.
                if expected_size is not None:
                    if downloaded_size != int(expected_size):
                        raise IOError(
                            f"Incomplete download: expected {expected_size} bytes, "
                            f"received {downloaded_size} bytes."
                        )

            is_valid, validation_message = validate_pdf(temporary_path)

            if not is_valid:
                raise ValueError(
                    f"Downloaded file failed PDF validation: {validation_message}"
                )

            # Atomic replacement ensures an incomplete download never appears
            # under the final PDF filename.
            temporary_path.replace(destination)

            logger.info(
                "Download validated: %s (%d bytes)",
                destination.name,
                downloaded_size,
            )
            return True

        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            logger.warning(
                "Download attempt %d/%d failed for %s: %s",
                attempt,
                max_retries,
                destination.name,
                exc,
            )

            temporary_path.unlink(missing_ok=True)

            if attempt < max_retries:
                # Exponential backoff reduces repeated pressure on a server
                # that may be temporarily unavailable or rate-limiting requests.
                time.sleep(2 ** (attempt - 1))

        except Exception:
            # Unexpected failures should not be hidden as ordinary network
            # errors. Clean up the temporary file and let the caller see them.
            temporary_path.unlink(missing_ok=True)
            raise

    logger.error("All download attempts failed: %s", destination.name)
    return False


def build_paper_metadata(result: arxiv.Result, pdf_path: Path) -> dict[str, Any]:
    """Convert an arXiv result into the project's established metadata schema."""
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title,
        "authors": [author.name for author in result.authors],
        "abstract": result.summary,
        "published": result.published.isoformat(),
        "updated": result.updated.isoformat(),
        "categories": result.categories,
        "primary_category": result.primary_category,
        "pdf_url": result.pdf_url,
        "pdf_path": str(pdf_path.relative_to(PROJECT_ROOT)),
    }


def build_search_query(categories: list[str]) -> str:
    """Build an arXiv API query matching any of the requested categories."""
    category_terms = [f"cat:{category}" for category in categories]
    return "(" + " OR ".join(category_terms) + ")"


def fetch_and_process_papers(args: argparse.Namespace) -> int:
    """
    Search arXiv, download new papers, and update the persistent metadata file.

    Returns the number of successfully available papers processed during
    this run, including already-downloaded papers skipped as duplicates.
    """
    records = load_existing_metadata()
    existing_ids = get_existing_paper_ids(records)

    # Track results within this run as well, since a search can return more
    # than one version or overlapping category matches.
    seen_ids: set[str] = set()

    query = build_search_query(args.categories)
    sort_criterion = (
        arxiv.SortCriterion.SubmittedDate
        if args.sort_by == "submitted"
        else arxiv.SortCriterion.Relevance
    )

    search = arxiv.Search(
        query=query,
        max_results=args.max_results,
        sort_by=sort_criterion,
    )

    client = arxiv.Client(
        page_size=min(args.max_results, 100),
        delay_seconds=3.0,
        num_retries=3,
    )

    processed_count = 0
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    logger.info("Starting arXiv search.")
    logger.info("Categories: %s", ", ".join(args.categories))
    logger.info("Maximum search results: %d", args.max_results)
    logger.info("Query: %s", query)

    for result in client.results(search):
        arxiv_id = result.get_short_id()
        base_id = get_base_arxiv_id(arxiv_id)

        if base_id in seen_ids:
            logger.info("Skipping duplicate search result: %s", arxiv_id)
            continue

        seen_ids.add(base_id)

        logger.info("")
        logger.info("Processing: %s", result.title)
        logger.info("arXiv ID: %s", arxiv_id)

        # A matching local record means this paper is already in the corpus.
        if base_id in existing_ids:
            logger.info("Already present in corpus; skipping download.")
            skipped_count += 1
            processed_count += 1
            continue

        pdf_path = RAW_DIR / f"{arxiv_id}.pdf"

        # Recover gracefully if a PDF exists but its metadata was not saved
        # during an earlier interrupted run.
        if pdf_path.exists():
            is_valid, message = validate_pdf(pdf_path)

            if is_valid:
                logger.info("Valid PDF already exists; reusing it.")
            else:
                logger.warning(
                    "Existing PDF is invalid and will be replaced: %s",
                    message,
                )
                pdf_path.unlink(missing_ok=True)

        if not pdf_path.exists():
            if not download_pdf(
                result.pdf_url,
                pdf_path,
                args.max_download_retries,
            ):
                failed_count += 1
                continue

            downloaded_count += 1

        metadata = build_paper_metadata(result, pdf_path)
        records.append(metadata)

        # Persist after each successful paper. A later failure will not
        # discard metadata for papers that have already been downloaded.
        save_metadata_atomically(records)

        existing_ids.add(base_id)
        processed_count += 1

        logger.info("Successfully processed: %s", pdf_path.name)

        # Avoid making consecutive PDF requests too aggressively.
        time.sleep(DOWNLOAD_DELAY_SECONDS)

    logger.info("")
    logger.info("=" * 60)
    logger.info("ARXIV INGESTION SUMMARY")
    logger.info("=" * 60)
    logger.info("Successfully processed: %d", processed_count)
    logger.info("Newly downloaded: %d", downloaded_count)
    logger.info("Skipped (already present): %d", skipped_count)
    logger.info("Failed downloads: %d", failed_count)
    logger.info("Total metadata records: %d", len(records))
    logger.info("Metadata file: %s", METADATA_FILE)
    logger.info("=" * 60)

    return failed_count


def main() -> int:
    """Run the ingestion pipeline and return a process exit status."""
    configure_logging()
    args = parse_arguments()

    try:
        configure_directories()
        failed_count = fetch_and_process_papers(args)

    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user.")
        return 130

    except Exception:
        logger.exception("Ingestion failed due to an unexpected error.")
        return 1

    return 1 if failed_count else 0


if __name__ == "__main__":
    sys.exit(main())
