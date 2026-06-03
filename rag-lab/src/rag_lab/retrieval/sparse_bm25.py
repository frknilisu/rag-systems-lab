"""BM25 sparse retriever — keyword-based search over the indexed corpus.

BM25 (Best Match 25) is a classical probabilistic ranking function.
Unlike dense retrieval it works on exact token matches, making it excellent
for rare words, product codes, proper nouns, and anything an embedding model
might generalise away.

The index is the .pkl file written by Indexer._rebuild_bm25().

n8n node mapping: "BM25 Search" Function node (reads the serialised index,
scores against the query tokens, returns top-k IDs to a merge node).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import structlog

from rag_lab.core.errors import RetrievalError
from rag_lab.core.types import Chunk, RetrievalResult

log = structlog.get_logger(__name__)


class BM25Retriever:
    """Load the persisted BM25 index and search it by token overlap.

    BM25 scores rise with term frequency in the document (TF) and fall with
    how common the term is across the whole corpus (IDF). It's fast, has no
    inference cost, and never hallucinates a match.

    Args:
        index_path: Path to the pickle file written by Indexer.
    """

    def __init__(self, index_path: str | Path = ".bm25_index.pkl") -> None:
        self._index_path = Path(index_path)
        self._bm25: object | None = None
        self._chunks: list[Chunk] = []

    # ── Public ────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True if an index file exists and can be loaded."""
        return self._index_path.exists()

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Score all indexed chunks against the query tokens, return top-k.

        Args:
            query: Natural-language query (tokenised by whitespace).
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievalResult with BM25 scores, sorted descending.
            Returns an empty list (not an error) if the index doesn't exist yet.
        """
        if not self.is_available():
            log.warning("bm25_index_missing", path=str(self._index_path))
            return []

        self._load_if_needed()
        tokens = query.lower().split()

        try:
            import numpy as np
            scores: list[float] = self._bm25.get_scores(tokens)  # type: ignore[union-attr]
            top_indices = np.argsort(scores)[::-1][:top_k]
        except Exception as exc:
            raise RetrievalError(f"BM25 scoring failed: {exc}") from exc

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score <= 0.0:
                break  # remaining chunks have no overlap with query tokens
            results.append(
                RetrievalResult(
                    chunk=self._chunks[idx],
                    score=score,
                    rank=rank,
                )
            )

        log.debug(
            "bm25_retrieved",
            query=query[:60],
            top_k=top_k,
            n_results=len(results),
            top_score=results[0].score if results else None,
        )
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_if_needed(self) -> None:
        if self._bm25 is not None:
            return
        try:
            with self._index_path.open("rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._chunks = data["chunks"]
            log.debug("bm25_index_loaded", n_chunks=len(self._chunks))
        except Exception as exc:
            raise RetrievalError(
                f"Failed to load BM25 index from {self._index_path}: {exc}"
            ) from exc

    def invalidate_cache(self) -> None:
        """Force a reload from disk on the next retrieve() call."""
        self._bm25 = None
        self._chunks = []
