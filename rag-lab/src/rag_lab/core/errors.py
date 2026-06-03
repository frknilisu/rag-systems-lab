"""Typed exception hierarchy for RAG-Lab.

Raise specific sub-types so callers can catch exactly what they care about
without swallowing unrelated exceptions.
"""

from __future__ import annotations


class RagLabError(Exception):
    """Base class for all RAG-Lab errors."""


class ConfigError(RagLabError):
    """Raised when configuration is invalid or missing."""


class ProviderError(RagLabError):
    """Raised when an external provider (LLM, embedder, vector store) fails."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class RetrievalError(RagLabError):
    """Raised when document retrieval fails."""


class GenerationError(RagLabError):
    """Raised when answer generation fails."""


class IndexError(RagLabError):
    """Raised when document indexing fails."""
