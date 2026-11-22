
"""
Main application entry point for the SciLitRAG API.

This module configures application logging, suppresses generic
Transformers progress bars, and registers the API route modules.
"""

# ---------------------------------------------------------------------------
# CPU Thread Configuration
# ---------------------------------------------------------------------------
from __future__ import annotations
import logging
import os

# Limit native numerical-library thread pools before importing PyTorch,
# NumPy, SciPy, or Sentence Transformers.
#
# These limits reduce CPU oversubscription during retrieval and reranking.
# They do not change model precision, retrieval parameters, or GPU placement.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import torch

# Configure PyTorch CPU execution before model initialization or inference.
torch.set_num_threads(8)
torch.set_num_interop_threads(4)

from transformers.utils import logging as transformers_logging

# Configure a dedicated console handler for SciLitRAG.
# Uvicorn's default logging configuration does not guarantee that
# application package loggers have a console handler.
scilitrag_logger = logging.getLogger("scilitrag")
scilitrag_logger.setLevel(logging.INFO)

if not scilitrag_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    scilitrag_logger.addHandler(console_handler)

# Prevent duplicate messages through the root logger.
scilitrag_logger.propagate = False

# Use named SciLitRAG model initialization logs instead of generic
# Transformers loading progress bars.
transformers_logging.disable_progress_bar()

from fastapi import FastAPI

from scilitrag.api.routes import ask, health, papers, compare


# Create the central FastAPI application.
app = FastAPI(
    title="SciLitRAG API",
    description=(
        "A research-paper assistant powered by hybrid retrieval, "
        "cross-encoder reranking, and locally hosted language models."
    ),
    version="0.1.0",
)

# Register health-check endpoints.
app.include_router(
    health.router,
    tags=["Health"],
)

# Register research-question endpoints exactly once.
app.include_router(
    ask.router,
    prefix="/api/v1",
    tags=["Research"],
)

# Register paper-library endpoints.
app.include_router(
    papers.router,
    prefix="/api/v1",
    tags=["Research"],
)

# Register paper-comparison endpoints.
app.include_router(
    compare.router,
    prefix="/api/v1",
    tags=["Research"],
)