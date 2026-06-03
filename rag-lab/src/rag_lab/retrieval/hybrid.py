"""Hybrid retriever — dense + sparse search fused with RRF.

Dense retrieval finds semantically similar chunks; BM25 finds exact keyword
matches. They fail in complementary ways — combining them with Reciprocal
Rank Fusion almost always beats either alone.

This class is the main building block of Hybrid RAG (Phase 3).

n8n node mapping: two parallel branches (Dense Search, BM25 Search) feeding
a single "RRF Merge" node.
"""

from __future__ import annotations

import structlog

from rag_lab.retrieval.dense import DenseRetriever
from rag_lab.retrieval.fusion import reciprocal_rank_fusion
from rag_lab.retrieval.sparse_bm25 import BM25Retriever
from rag_lab.core.types import RetrievalResult

log = structlog.get_logger(__name__)


class HybridRetriever:
    """Run dense and BM25 retrievers in parallel, fuse results with RRF.

    Args:
        dense: Dense retriever (vector similarity).
        sparse: BM25 keyword retriever.
        top_k: Final number of results to return after fusion.
        rrf_k: RRF smoothing constant (default 60).
        candidate_multiplier: Each retriever fetches top_k × this many candidates
            before fusion so there's enough overlap to re-rank meaningfully.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        top_k: int = 5,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Return top_k chunks fused from dense and BM25 results."""
        candidates = self.top_k * self.candidate_multiplier

        dense_results = self.dense.retrieve(query, top_k=candidates)
        sparse_results = self.sparse.retrieve(query, top_k=candidates)

        if not sparse_results:
            log.debug("hybrid_fallback_dense_only", reason="bm25_unavailable")
            return dense_results[: self.top_k]

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
            top_k=self.top_k,
        )
        log.debug(
            "hybrid_retrieved",
            query=query[:60],
            n_dense=len(dense_results),
            n_sparse=len(sparse_results),
            n_fused=len(fused),
        )
        return fused
