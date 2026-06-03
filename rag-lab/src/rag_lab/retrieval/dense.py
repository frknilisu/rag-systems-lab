"""Dense retriever — embed the query, search the vector store.

This is the retrieval half of every RAG pipeline: given a natural-language
question, find the k most semantically similar chunks.

n8n node mapping: two nodes — "Embed Query" (Function) + "Vector Search" (Chroma node).
"""

from __future__ import annotations

import structlog

from rag_lab.core.interfaces import EmbeddingProvider, VectorStore
from rag_lab.core.types import RetrievalResult

log = structlog.get_logger(__name__)


class DenseRetriever:
    """Embed the query, run cosine-similarity search, return top-k results.

    This is the baseline retriever used by Standard RAG (Phase 2). Dense
    retrieval finds *semantically* similar chunks even when the exact words
    don't match — the embedding model handles synonyms and paraphrases.

    Limitation: it can miss chunks that match on rare exact keywords (product
    codes, IDs, names) when the embedding model's training data didn't cover
    those terms. That's where BM25 helps (Hybrid RAG, Phase 3).
    """

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[RetrievalResult]:
        """Embed the query and return the top-k most similar chunks.

        Args:
            query: Natural-language question.
            top_k: Number of results to return.
            filter: Optional metadata filter forwarded to the vector store.

        Returns:
            List of RetrievalResult, sorted by descending similarity score.
        """
        query_vector = self.embedder.embed_one(query)
        results = self.store.search(query_vector, top_k=top_k, filter=filter)
        log.debug(
            "dense_retrieved",
            query=query[:60],
            top_k=top_k,
            n_results=len(results),
            top_score=results[0].score if results else None,
        )
        return results
