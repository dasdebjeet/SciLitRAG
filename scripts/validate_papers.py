
"""
Validate downloaded research papers before PDF text extraction.

Checks:
    1. Metadata JSON exists and contains valid records.
    2. Every referenced PDF exists.
    3. Each PDF can be opened by PyMuPDF.
    4. Each PDF contains at least one page.
    5. Text extraction produces a meaningful amount of text.

Exit codes:
    0: All PDFs passed structural validation.
    1: One or more PDFs failed validation.
"""

import json
import logging
import sys
from pathlib import Path

import pymupdf as fitz


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "papers.json"

# PDFs containing less text may be scanned documents or image-only files.
# We report these separately rather than automatically treating them as corrupt.
MIN_EXTRACTED_CHARACTERS = 100

# Configure application-level logging for readable validation reports.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def validate_pdf(pdf_path: Path) -> tuple[bool, int, int]:
    """
    Validate a PDF and inspect its extractable text.

    Args:
        pdf_path: Absolute path to the PDF being inspected.

    Returns:
        A tuple containing:
            - Whether structural validation passed.
            - Number of pages.
            - Number of extracted text characters.

    Raises:
        fitz.FileDataError: If PyMuPDF cannot interpret the PDF.
        RuntimeError: If the PDF is encrypted and cannot be inspected.
    """

    with fitz.open(pdf_path) as document:

        # Reject encrypted documents because we do not have their passwords.
        if document.is_encrypted:
            raise RuntimeError("PDF is encrypted and cannot be inspected.")

        page_count = len(document)

        if page_count == 0:
            raise RuntimeError("PDF contains no pages.")

        # Extract text from every page to check whether the document
        # is suitable for our downstream text-processing pipeline.
        extracted_text = "\n".join(
            page.get_text("text") for page in document
        )

        character_count = len(extracted_text.strip())

        return True, page_count, character_count


def main() -> int:
    """
    Validate all PDFs referenced in the paper metadata.

    Returns:
        Process exit code: 0 for success, 1 if validation failures occur.
    """

    if not METADATA_PATH.exists():
        logger.error("Metadata file not found: %s", METADATA_PATH)
        return 1

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            papers = json.load(file)

    except (json.JSONDecodeError, OSError) as error:
        logger.error("Could not read metadata: %s", error)
        return 1

    if not isinstance(papers, list) or not papers:
        logger.error("Metadata must contain a non-empty list of papers.")
        return 1

    passed = 0
    failed = 0
    low_text = 0

    logger.info("Starting validation of %d papers.", len(papers))

    for paper in papers:
        title = paper.get("title", "Unknown title")
        relative_path = paper.get("pdf_path")

        if not relative_path:
            logger.error("Missing PDF path for paper: %s", title)
            failed += 1
            continue

        pdf_path = PROJECT_ROOT / relative_path

        if not pdf_path.is_file():
            logger.error("PDF not found: %s", pdf_path)
            failed += 1
            continue

        try:
            _, page_count, character_count = validate_pdf(pdf_path)

            if character_count < MIN_EXTRACTED_CHARACTERS:
                logger.warning(
                    "LOW TEXT | %s | Pages: %d | Characters: %d",
                    title,
                    page_count,
                    character_count,
                )
                low_text += 1
            else:
                logger.info(
                    "VALID | %s | Pages: %d | Characters: %d",
                    title,
                    page_count,
                    character_count,
                )

            passed += 1

        except (RuntimeError, fitz.FileDataError, OSError) as error:
            logger.error("INVALID | %s | Reason: %s", title, error)
            failed += 1

    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("Total papers: %d", len(papers))
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("Low-text PDFs: %d", low_text)
    logger.info("=" * 60)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
