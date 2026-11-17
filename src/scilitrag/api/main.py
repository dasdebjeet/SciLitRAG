"""
Main application entry point for the SciLitRAG API.

This module creates the FastAPI application and registers all
available route modules.
"""

from __future__ import annotations

from fastapi import FastAPI

from scilitrag.api.routes import ask, health


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

# Register research-question endpoints under a versioned API prefix.
app.include_router(
    ask.router,
    prefix="/api/v1",
    tags=["Research"],
)