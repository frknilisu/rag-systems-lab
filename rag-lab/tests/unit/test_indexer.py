"""Tests for ingestion/indexer.py — uses fake providers, no real I/O."""

from __future__ import annotations

import pickle

import pytest

from rag_lab.core.types import Document
from rag_lab.ingestion.chunkers import RecursiveChunker
from rag_lab.ingestion.indexer import Indexer


def _make_indexer(fake_embedder, fake_store, tmp_path):
    return Indexer(
        embedder=fake_embedder,
        store=fake_store,
        chunker=RecursiveChunker(chunk_size=200, chunk_overlap=20),
        bm25_index_path=str(tmp_path / "bm25.pkl"),
    )


def test_index_stores_chunks_in_vector_store(fake_embedder, fake_store, tmp_path):
    indexer = _make_indexer(fake_embedder, fake_store, tmp_path)
    docs = [Document(id="d1", text="Paris is the capital of France. " * 10)]
    stats = indexer.index(docs)
    assert stats.num_documents == 1
    assert stats.num_chunks >= 1
    assert fake_store.count() >= 1


def test_index_returns_correct_stats(fake_embedder, fake_store, tmp_path):
    indexer = _make_indexer(fake_embedder, fake_store, tmp_path)
    docs = [
        Document(id="d1", text="Short doc."),
        Document(id="d2", text="Another short doc."),
    ]
    stats = indexer.index(docs)
    assert stats.num_documents == 2
    assert stats.num_chunks >= 2
    assert stats.duration_ms > 0


def test_index_builds_bm25_file(fake_embedder, fake_store, tmp_path):
    indexer = _make_indexer(fake_embedder, fake_store, tmp_path)
    indexer.index([Document(id="d1", text="Some text about retrieval.")])
    bm25_path = tmp_path / "bm25.pkl"
    assert bm25_path.exists()
    with bm25_path.open("rb") as f:
        data = pickle.load(f)
    assert "bm25" in data
    assert "chunks" in data
    assert len(data["chunks"]) >= 1


def test_incremental_indexing_merges_chunks(fake_embedder, fake_store, tmp_path):
    indexer = _make_indexer(fake_embedder, fake_store, tmp_path)
    indexer.index([Document(id="d1", text="First document.")])
    indexer.index([Document(id="d2", text="Second document.")])

    with (tmp_path / "bm25.pkl").open("rb") as f:
        data = pickle.load(f)
    chunk_ids = {c.id for c in data["chunks"]}
    # Both documents' chunks must be in the combined index
    assert any("d1" in cid for cid in chunk_ids)
    assert any("d2" in cid for cid in chunk_ids)


def test_reindex_same_doc_deduplicates(fake_embedder, fake_store, tmp_path):
    indexer = _make_indexer(fake_embedder, fake_store, tmp_path)
    doc = Document(id="d1", text="Same document indexed twice.")
    indexer.index([doc])
    count_after_first = fake_store.count()
    indexer.index([doc])
    # Chroma upserts by ID — count should not grow
    assert fake_store.count() == count_after_first


def test_empty_document_list_returns_zero_stats(fake_embedder, fake_store, tmp_path):
    indexer = _make_indexer(fake_embedder, fake_store, tmp_path)
    stats = indexer.index([])
    assert stats.num_documents == 0
    assert stats.num_chunks == 0


def test_index_file(fake_embedder, fake_store, tmp_path):
    indexer = _make_indexer(fake_embedder, fake_store, tmp_path)
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text("The quick brown fox jumps over the lazy dog.")
    stats = indexer.index_file(corpus_file)
    assert stats.num_documents == 1
    assert stats.num_chunks >= 1
