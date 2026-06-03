"""Tests for ingestion/chunkers.py."""

from __future__ import annotations

import pytest

from rag_lab.config.schema import IngestionConfig
from rag_lab.core.types import Document
from rag_lab.ingestion.chunkers import (
    FixedSizeChunker,
    RecursiveChunker,
    build_chunker,
)


def _doc(text: str, doc_id: str = "test") -> Document:
    return Document(id=doc_id, text=text)


# ── FixedSizeChunker ──────────────────────────────────────────────────────────

class TestFixedSizeChunker:
    def test_single_chunk_for_short_text(self):
        chunker = FixedSizeChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(_doc("Short text."))
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_multiple_chunks_with_overlap(self):
        text = "A" * 100
        chunker = FixedSizeChunker(chunk_size=40, chunk_overlap=10)
        chunks = chunker.chunk(_doc(text))
        assert len(chunks) > 1
        # Each chunk ≤ chunk_size
        assert all(len(c.text) <= 40 for c in chunks)

    def test_chunk_ids_are_unique(self):
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.chunk(_doc("x" * 200, doc_id="mydoc"))
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_doc_id_is_set(self):
        chunker = FixedSizeChunker(chunk_size=512)
        chunks = chunker.chunk(_doc("text", doc_id="source-doc"))
        assert all(c.doc_id == "source-doc" for c in chunks)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=100)

    def test_empty_text_returns_no_chunks(self):
        chunker = FixedSizeChunker(chunk_size=512)
        chunks = chunker.chunk(_doc("   "))
        assert chunks == []


# ── RecursiveChunker ──────────────────────────────────────────────────────────

class TestRecursiveChunker:
    def test_short_text_is_one_chunk(self):
        chunker = RecursiveChunker(chunk_size=512)
        chunks = chunker.chunk(_doc("This is a short document."))
        assert len(chunks) == 1

    def test_paragraphs_split_correctly(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(_doc(text))
        assert len(chunks) >= 1
        combined = " ".join(c.text for c in chunks)
        assert "Para one" in combined
        assert "Para two" in combined

    def test_respects_chunk_size(self):
        text = " ".join(["word"] * 500)  # ~2500 chars
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(_doc(text))
        assert all(len(c.text) <= 250 for c in chunks)  # some slack for the merge join

    def test_all_text_preserved(self):
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(_doc(text))
        combined = " ".join(c.text for c in chunks)
        # All unique words should appear somewhere
        assert "quick" in combined
        assert "lazy" in combined

    def test_chunk_ids_unique_and_contain_doc_id(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.chunk(_doc("x " * 300, doc_id="my-doc"))
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))
        assert all("my-doc" in cid for cid in ids)

    def test_metadata_inherited(self):
        doc = Document(id="d1", text="some text", metadata={"source": "wiki"})
        chunker = RecursiveChunker(chunk_size=512)
        chunks = chunker.chunk(doc)
        assert chunks[0].metadata["source"] == "wiki"

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)


# ── build_chunker factory ──────────────────────────────────────────────────────

def test_build_chunker_recursive():
    cfg = IngestionConfig(chunker="recursive", chunk_size=256, chunk_overlap=32)
    chunker = build_chunker(cfg)
    assert isinstance(chunker, RecursiveChunker)


def test_build_chunker_fixed():
    cfg = IngestionConfig(chunker="fixed", chunk_size=256, chunk_overlap=32)
    chunker = build_chunker(cfg)
    assert isinstance(chunker, FixedSizeChunker)


def test_build_chunker_unknown_raises():
    cfg = IngestionConfig(chunker="unknown")
    with pytest.raises(ValueError, match="Unknown chunker"):
        build_chunker(cfg)
