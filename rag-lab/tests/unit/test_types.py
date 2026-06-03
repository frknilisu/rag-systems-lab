"""Tests for core/types.py — the shared data contract."""

from __future__ import annotations

import json

import pytest

from rag_lab.core.types import (
    Chunk,
    Document,
    IndexStats,
    RAGResult,
    RAGState,
    RetrievalResult,
    StepTrace,
)


def test_document_round_trips_json():
    doc = Document(id="d1", text="hello", metadata={"k": "v"})
    restored = Document.model_validate_json(doc.model_dump_json())
    assert restored == doc


def test_chunk_has_optional_embedding():
    c = Chunk(id="c1", doc_id="d1", text="chunk text")
    assert c.embedding is None

    c_with_emb = Chunk(id="c1", doc_id="d1", text="chunk text", embedding=[0.1, 0.2])
    assert c_with_emb.embedding == [0.1, 0.2]


def test_retrieval_result_defaults():
    chunk = Chunk(id="c1", doc_id="d1", text="test")
    result = RetrievalResult(chunk=chunk, score=0.95)
    assert result.rank == 0


def test_rag_state_is_json_serializable():
    state = RAGState(question="What is RAG?", session_id="sess-1")
    raw = state.model_dump_json()
    data = json.loads(raw)
    assert data["question"] == "What is RAG?"
    assert data["retrieval_results"] == []


def test_rag_state_collects_results(sample_chunks):
    state = RAGState(question="test")
    state.retrieval_results = [
        RetrievalResult(chunk=sample_chunks[0], score=0.9, rank=0),
        RetrievalResult(chunk=sample_chunks[1], score=0.7, rank=1),
    ]
    assert len(state.retrieval_results) == 2
    assert state.retrieval_results[0].score == 0.9


def test_rag_result_empty_metrics():
    result = RAGResult(answer="answer", contexts=[], trace=[])
    assert result.metrics == {}
    assert result.config_snapshot == {}


def test_step_trace_has_timestamps():
    trace = StepTrace(step="retrieve", latency_ms=12.5)
    assert trace.started_at is not None
    assert trace.latency_ms == 12.5


def test_index_stats():
    stats = IndexStats(num_documents=3, num_chunks=9, duration_ms=150.0)
    assert stats.num_chunks == 9
