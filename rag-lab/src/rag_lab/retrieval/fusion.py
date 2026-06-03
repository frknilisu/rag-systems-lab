"""Reciprocal Rank Fusion (RRF) — combine ranked lists from multiple retrievers.

RRF was introduced by Cormack et al. (2009) as a simple, parameter-light way
to merge multiple ranked lists. The key insight: *rank* matters more than
*score*, and contributions from each list decay as 1 / (k + rank).

This makes it robust to score-scale differences between dense (cosine 0–1)
and sparse (BM25, unbounded) retrievers — you never need to normalise scores.

Reference: Cormack, Clarke, Buettcher. "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods." SIGIR 2009.

n8n node mapping: "Merge & Re-rank" Function node that receives two arrays
(dense results, sparse results) and returns one fused array.
"""

from __future__ import annotations

from rag_lab.core.types import Chunk, RetrievalResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Each chunk accumulates a score of  1 / (k + rank)  for each list it
    appears in. Chunks that appear high in multiple lists score highest.

    Args:
        ranked_lists: One ranked list per retriever (dense, sparse, …).
        k: Smoothing constant (default 60, from the original paper).
           Higher k → flatter contribution curve; lower k → top ranks dominate.
        top_k: If set, truncate the result to this many items.

    Returns:
        Single fused list sorted by descending RRF score.
    """
    rrf_scores: dict[str, float] = {}
    chunk_registry: dict[str, Chunk] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list):
            chunk_id = result.chunk.id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_registry[chunk_id] = result.chunk

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    if top_k is not None:
        fused = fused[:top_k]

    return [
        RetrievalResult(chunk=chunk_registry[cid], score=score, rank=rank)
        for rank, (cid, score) in enumerate(fused)
    ]
