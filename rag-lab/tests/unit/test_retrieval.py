"""Tests for retrieval/: dense, BM25, fusion, hybrid."""

from __future__ import annotations

import pickle

import pytest

from rag_lab.core.types import Chunk, Document, RetrievalResult
from rag_lab.ingestion.chunkers import RecursiveChunker
from rag_lab.ingestion.indexer import Indexer
from rag_lab.retrieval.dense import DenseRetriever
from rag_lab.retrieval.fusion import reciprocal_rank_fusion
from rag_lab.retrieval.hybrid import HybridRetriever
from rag_lab.retrieval.sparse_bm25 import BM25Retriever


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_index(fake_embedder, fake_store, tmp_path, docs=None):
    """Index sample docs; return (indexer, bm25_path)."""
    if docs is None:
        docs = [
            Document(id="d1", text="Paris is the capital of France."),
            Document(id="d2", text="Berlin is the capital of Germany."),
            Document(id="d3", text="Tokyo is the capital of Japan."),
        ]
    bm25_path = tmp_path / "bm25.pkl"
    indexer = Indexer(
        embedder=fake_embedder,
        store=fake_store,
        chunker=RecursiveChunker(chunk_size=200, chunk_overlap=20),
        bm25_index_path=str(bm25_path),
    )
    indexer.index(docs)
    return indexer, str(bm25_path)


# ── DenseRetriever ─────────────────────────────────────────────────────────────

class TestDenseRetriever:
    def test_returns_results(self, fake_embedder, fake_store, tmp_path):
        _build_index(fake_embedder, fake_store, tmp_path)
        retriever = DenseRetriever(fake_embedder, fake_store)
        results = retriever.retrieve("What is the capital of France?", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_results_have_scores(self, fake_embedder, fake_store, tmp_path):
        _build_index(fake_embedder, fake_store, tmp_path)
        retriever = DenseRetriever(fake_embedder, fake_store)
        results = retriever.retrieve("capital", top_k=3)
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_empty_store_returns_empty(self, fake_embedder, fake_store):
        retriever = DenseRetriever(fake_embedder, fake_store)
        results = retriever.retrieve("anything", top_k=5)
        assert results == []


# ── BM25Retriever ──────────────────────────────────────────────────────────────

class TestBM25Retriever:
    def test_not_available_without_index(self, tmp_path):
        retriever = BM25Retriever(tmp_path / "nonexistent.pkl")
        assert not retriever.is_available()

    def test_returns_empty_when_unavailable(self, tmp_path):
        retriever = BM25Retriever(tmp_path / "nonexistent.pkl")
        results = retriever.retrieve("query")
        assert results == []

    def test_retrieves_after_indexing(self, fake_embedder, fake_store, tmp_path):
        _, bm25_path = _build_index(fake_embedder, fake_store, tmp_path)
        retriever = BM25Retriever(bm25_path)
        assert retriever.is_available()
        results = retriever.retrieve("capital France", top_k=3)
        assert isinstance(results, list)

    def test_scores_are_positive(self, fake_embedder, fake_store, tmp_path):
        _, bm25_path = _build_index(fake_embedder, fake_store, tmp_path)
        retriever = BM25Retriever(bm25_path)
        results = retriever.retrieve("capital", top_k=5)
        assert all(r.score > 0 for r in results)

    def test_ranks_are_sequential(self, fake_embedder, fake_store, tmp_path):
        _, bm25_path = _build_index(fake_embedder, fake_store, tmp_path)
        retriever = BM25Retriever(bm25_path)
        results = retriever.retrieve("capital Germany", top_k=3)
        assert [r.rank for r in results] == list(range(len(results)))

    def test_invalidate_cache_forces_reload(self, fake_embedder, fake_store, tmp_path):
        _, bm25_path = _build_index(fake_embedder, fake_store, tmp_path)
        retriever = BM25Retriever(bm25_path)
        retriever.retrieve("query", top_k=1)   # loads cache
        retriever.invalidate_cache()
        assert retriever._bm25 is None


# ── reciprocal_rank_fusion ─────────────────────────────────────────────────────

def _make_result(chunk_id: str, score: float = 1.0, rank: int = 0) -> RetrievalResult:
    chunk = Chunk(id=chunk_id, doc_id="doc", text=f"text-{chunk_id}")
    return RetrievalResult(chunk=chunk, score=score, rank=rank)


class TestRRF:
    def test_merges_two_lists(self):
        list_a = [_make_result("c1", rank=0), _make_result("c2", rank=1)]
        list_b = [_make_result("c2", rank=0), _make_result("c3", rank=1)]
        fused = reciprocal_rank_fusion([list_a, list_b])
        ids = [r.chunk.id for r in fused]
        assert set(ids) == {"c1", "c2", "c3"}

    def test_overlap_boosts_score(self):
        list_a = [_make_result("shared", rank=0), _make_result("only_a", rank=1)]
        list_b = [_make_result("shared", rank=0), _make_result("only_b", rank=1)]
        fused = reciprocal_rank_fusion([list_a, list_b])
        # "shared" appears at rank 0 in both — must be top result
        assert fused[0].chunk.id == "shared"

    def test_fused_ranks_are_sequential(self):
        list_a = [_make_result(f"c{i}", rank=i) for i in range(5)]
        fused = reciprocal_rank_fusion([list_a])
        assert [r.rank for r in fused] == list(range(len(fused)))

    def test_top_k_truncates(self):
        list_a = [_make_result(f"c{i}", rank=i) for i in range(10)]
        fused = reciprocal_rank_fusion([list_a], top_k=3)
        assert len(fused) == 3

    def test_empty_lists_return_empty(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_single_list_preserves_order(self):
        ranked = [_make_result(f"c{i}", rank=i) for i in range(4)]
        fused = reciprocal_rank_fusion([ranked])
        assert [r.chunk.id for r in fused] == [r.chunk.id for r in ranked]


# ── HybridRetriever ────────────────────────────────────────────────────────────

class TestHybridRetriever:
    def test_falls_back_to_dense_when_bm25_unavailable(
        self, fake_embedder, fake_store, tmp_path
    ):
        _build_index(fake_embedder, fake_store, tmp_path)
        dense = DenseRetriever(fake_embedder, fake_store)
        sparse = BM25Retriever(tmp_path / "nonexistent.pkl")
        hybrid = HybridRetriever(dense, sparse, top_k=2)
        results = hybrid.retrieve("capital")
        assert isinstance(results, list)

    def test_combines_both_retrievers(self, fake_embedder, fake_store, tmp_path):
        _, bm25_path = _build_index(fake_embedder, fake_store, tmp_path)
        dense = DenseRetriever(fake_embedder, fake_store)
        sparse = BM25Retriever(bm25_path)
        hybrid = HybridRetriever(dense, sparse, top_k=3)
        results = hybrid.retrieve("capital of Japan")
        assert isinstance(results, list)
        assert len(results) <= 3
