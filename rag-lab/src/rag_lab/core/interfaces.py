"""Provider interfaces — defined as typing.Protocol so implementations need not subclass.

Every concrete provider (LiteLLM, SentenceTransformers, Chroma, …) is registered
via the registry and injected through config; no architecture ever imports a
provider directly.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from rag_lab.core.types import Chunk, RetrievalResult


@runtime_checkable
class LLMProvider(Protocol):
    """Synchronous LLM text completion."""

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """Send a chat-formatted message list; return the assistant reply."""
        ...

    def stream(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> Iterator[str]:
        """Stream the assistant reply token by token."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Dense text embedding."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; return one float vector per text."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for a single string."""
        ...

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Dense nearest-neighbour store."""

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, object]],
    ) -> None:
        """Insert or update vectors with associated payloads."""
        ...

    def search(
        self,
        vector: list[float],
        top_k: int,
        filter: dict[str, object] | None = None,
    ) -> list[RetrievalResult]:
        """Return the top-k most similar chunks."""
        ...

    def delete(self, ids: list[str]) -> None:
        """Remove vectors by id."""
        ...

    def count(self) -> int:
        """Number of vectors currently stored."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder reranker that re-scores an initial retrieval list."""

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Return results re-ordered by cross-encoder score (highest first)."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Knowledge-graph store (used by Graph-Augmented RAG)."""

    def add_node(self, node_id: str, **attrs: object) -> None: ...
    def add_edge(self, src: str, dst: str, **attrs: object) -> None: ...
    def neighbors(self, node_id: str, hops: int = 1) -> list[str]: ...


@runtime_checkable
class MemoryStore(Protocol):
    """Session-scoped memory for Memory-Augmented RAG."""

    def save(self, session_id: str, chunk: Chunk) -> None: ...
    def load(self, session_id: str, top_k: int = 5) -> list[Chunk]: ...
    def clear(self, session_id: str) -> None: ...
