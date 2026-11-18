"""Centralized filesystem paths for the SciLitRAG project."""

from pathlib import Path

# Resolve the repository root from this file's location:
# src/scilitrag/core/paths.py -> repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
EVALUATION_DIR = DATA_DIR / "evaluation"

DEFAULT_CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"

# Metadata describing the papers available in the local research library.
PAPERS_METADATA_PATH = METADATA_DIR / "papers.json"
