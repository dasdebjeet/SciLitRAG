
"""
SciLitRAG - PDF Text Extraction Pipeline

Purpose:
    Extract page-level text from downloaded arXiv research papers
    while preserving document metadata and source provenance.

Input:
    data/metadata/papers.json
    data/raw/*.pdf

Output:
    data/processed/pages.jsonl

Each JSONL record represents one PDF page and includes:
    - arXiv ID
    - Paper title
    - PDF path
    - Page number (1-indexed)
    - Total page count
    - Extracted text
    - Character count
    - Original PDF metadata

Design considerations:
    - Processes one PDF at a time to control memory usage.
    - Uses temporary output files to avoid replacing valid output
      with an incomplete extraction.
    - Continues processing if an individual PDF fails.
    - Preserves page-level provenance for downstream citations.

Limitations:
    PyMuPDF extracts embedded PDF text. It does not perform OCR
    on scanned or image-only pages.
"""

import json
import logging
import sys

from pathlib import Path
from typing import Any

import pymupdf as fitz


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "papers.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "pages.jsonl"

# Temporary output is used so that an interrupted extraction does
# not overwrite the previous successfully generated dataset.
TEMP_OUTPUT_PATH = PROCESSED_DIR / "pages.jsonl.tmp"


# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------

def load_paper_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    """
    Load and validate the metadata generated during arXiv ingestion.

    Args:
        metadata_path: Path to the papers.json metadata file.

    Returns:
        A list containing one metadata dictionary per paper.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
        ValueError: If the JSON structure is invalid.
    """

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Paper metadata file not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        papers = json.load(file)

    if not isinstance(papers, list):
        raise ValueError("Paper metadata must contain a JSON list.")

    if not papers:
        raise ValueError("Paper metadata contains no records.")

    # Validate required fields before starting potentially expensive
    # PDF processing.
    for index, paper in enumerate(papers):
        if not isinstance(paper, dict):
            raise ValueError(f"Invalid paper record at index {index}.")

        required_fields = ("arxiv_id", "title", "pdf_path")

        missing_fields = [
            field for field in required_fields if not paper.get(field)
        ]

        if missing_fields:
            raise ValueError(
                f"Paper record {index} is missing fields: {missing_fields}"
            )

    return papers


# ---------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------

def extract_pdf_pages(
    paper: dict[str, Any],
    project_root: Path,
) -> list[dict[str, Any]]:
    """
    Extract text from every page of a research paper.

    Args:
        paper: Metadata dictionary associated with the PDF.
        project_root: Absolute path to the project root directory.

    Returns:
        A list of page-level records containing text and provenance.

    Raises:
        FileNotFoundError: If the referenced PDF does not exist.
        RuntimeError: If the PDF is encrypted or contains no pages.
        fitz.FileDataError: If PyMuPDF cannot interpret the PDF.
    """

    # Resolve relative paths against the project root, keeping the
    # metadata independent of the machine on which it was generated.
    pdf_path = (project_root / paper["pdf_path"]).resolve()

    if not pdf_path.is_relative_to(project_root.resolve()):
        raise ValueError("PDF path resolves outside the project directory.")

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    page_records = []

    with fitz.open(pdf_path) as document:

        if document.is_encrypted:
            raise RuntimeError("Encrypted PDF cannot be processed.")

        page_count = len(document)

        if page_count == 0:
            raise RuntimeError("PDF contains no pages.")

        # Preserve the original PDF metadata when available.
        pdf_metadata = document.metadata or {}

        for page_index, page in enumerate(document):

            # sort=True attempts to return text in natural reading
            # order rather than the PDF's internal object order.
            extracted_text = page.get_text(
                "text",
                sort=True,
            )

            # Strip surrounding whitespace without aggressively
            # modifying the extracted research content.
            extracted_text = extracted_text.strip()

            page_record = {
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "pdf_path": paper["pdf_path"],
                "page_number": page_index + 1,
                "total_pages": page_count,
                "text": extracted_text,
                "character_count": len(extracted_text),
                "pdf_metadata": pdf_metadata,
            }

            page_records.append(page_record)

    return page_records


# ---------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------

def main() -> int:
    """
    Execute the PDF extraction pipeline for the complete paper corpus.

    Returns:
        0 if all PDFs were processed successfully.
        1 if one or more PDFs could not be processed.
    """

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        papers = load_paper_metadata(METADATA_PATH)

    except (OSError, ValueError, json.JSONDecodeError) as error:
        logger.exception("Unable to load paper metadata: %s", error)
        return 1

    total_pages = 0
    successful_papers = 0
    failed_papers = 0
    empty_pages = 0

    logger.info("Starting extraction for %d papers.", len(papers))

    try:
        # Write the new dataset to a temporary file first.
        # The final output is replaced only after processing completes.
        with TEMP_OUTPUT_PATH.open("w", encoding="utf-8") as output_file:

            for paper in papers:

                paper_id = paper["arxiv_id"]

                logger.info(
                    "Extracting paper: %s | %s",
                    paper_id,
                    paper["title"],
                )

                try:
                    page_records = extract_pdf_pages(
                        paper=paper,
                        project_root=PROJECT_ROOT,
                    )

                    # Write each page immediately instead of retaining
                    # the entire corpus in memory.
                    for record in page_records:

                        if record["character_count"] == 0:
                            empty_pages += 1

                            logger.warning(
                                "Empty text: %s | Page %d",
                                paper_id,
                                record["page_number"],
                            )

                        output_file.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                            ) + "\n"
                        )

                    total_pages += len(page_records)
                    successful_papers += 1

                    logger.info(
                        "Completed: %s | Pages: %d",
                        paper_id,
                        len(page_records),
                    )

                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    fitz.FileDataError,
                ) as error:

                    failed_papers += 1

                    logger.exception(
                        "Failed to extract paper %s: %s",
                        paper_id,
                        error,
                    )

        # Replace the previous output only after the new extraction
        # has completed without an unexpected pipeline-level failure.
        TEMP_OUTPUT_PATH.replace(OUTPUT_PATH)

    except OSError:
        logger.exception("Unable to write extraction output.")
        return 1

    logger.info("=" * 65)
    logger.info("PDF EXTRACTION SUMMARY")
    logger.info("Total papers: %d", len(papers))
    logger.info("Successfully processed: %d", successful_papers)
    logger.info("Failed papers: %d", failed_papers)
    logger.info("Total extracted pages: %d", total_pages)
    logger.info("Pages with empty text: %d", empty_pages)
    logger.info("Output: %s", OUTPUT_PATH)
    logger.info("=" * 65)

    return 1 if failed_papers else 0


if __name__ == "__main__":
    sys.exit(main())
