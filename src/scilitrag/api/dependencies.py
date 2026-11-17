
"""
Shared dependencies for the SciLitRAG API.

The answer pipeline is created once and reused across requests.
A process-wide lock serializes inference because the current pipeline
loads and releases multiple GPU models during each answer operation.
"""
from __future__ import annotations

from functools import lru_cache
from threading import Lock
from typing import Any



# One lock per API process. Only one request can execute GPU inference
# through this application at a time.
_INFERENCE_LOCK = Lock()


@lru_cache(maxsize=1)
def get_answer_pipeline() -> AnswerPipeline:
    """Return the shared, lightweight answer-pipeline instance."""

    from scilitrag.pipeline.answer_pipeline import AnswerPipeline
    
    return AnswerPipeline()


def get_inference_lock() -> Lock:
    """Return the lock protecting the GPU inference lifecycle."""

    return _INFERENCE_LOCK
