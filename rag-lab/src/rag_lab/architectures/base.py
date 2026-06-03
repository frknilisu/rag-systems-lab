"""Abstract base class for every RAG pipeline implementation.

All 10 architectures implement this interface so they are interchangeable
behind config. Use build_architecture() from the registry to get an instance.

n8n mapping: the RAGPipeline.query() method corresponds to the entire
             n8n workflow execution triggered by an HTTP webhook.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_lab.core.types import Document, IndexStats, RAGResult


class RAGPipeline(ABC):
    """Common contract for every architecture.

    Attributes:
        name: Short identifier used in config, CLI, and the registry
              (e.g. "standard", "hybrid", "hyde").
    """

    name: str

    @abstractmethod
    def index(self, documents: list[Document]) -> IndexStats:
        """Chunk, embed, and store a list of documents.

        Called once per corpus (or incrementally as new docs arrive).
        Returns statistics about what was indexed.
        """
        ...

    @abstractmethod
    def query(
        self,
        question: str,
        *,
        session_id: str | None = None,
    ) -> RAGResult:
        """Retrieve relevant context and generate an answer.

        Args:
            question: The user's natural-language question.
            session_id: Optional session identifier for memory-augmented
                        architectures; ignored by stateless ones.

        Returns:
            RAGResult with the answer, retrieved contexts, full step trace,
            and a snapshot of the config used.
        """
        ...
