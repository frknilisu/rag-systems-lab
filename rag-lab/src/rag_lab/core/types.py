"""Shared data types for the entire RAG-Lab system.

These Pydantic models form the contract between every layer: ingestion,
retrieval, generation, evaluation, and the n8n workflow blueprints.

RAGState is the "item" that flows through a pipeline — keep it JSON-serializable
so it maps directly onto n8n node outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Document & Chunk ──────────────────────────────────────────────────────────

class Document(BaseModel):
    """A raw source document before chunking."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A text chunk derived from a Document, ready for embedding and retrieval."""

    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


# ── Retrieval ─────────────────────────────────────────────────────────────────

class RetrievalResult(BaseModel):
    """A single retrieved chunk with its similarity score and rank."""

    chunk: Chunk
    score: float
    rank: int = 0


# ── Tracing ───────────────────────────────────────────────────────────────────

class StepTrace(BaseModel):
    """Timing and I/O record for one pipeline step.

    Kept cheap: inputs/outputs store only summary data, not full vectors.
    Doubles as a teaching artifact — docs show real traces.
    """

    step: str
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime = Field(default_factory=_utcnow)
    latency_ms: float = 0.0
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Pipeline state & result ───────────────────────────────────────────────────

class RAGState(BaseModel):
    """Mutable state object that flows through every pipeline step.

    JSON-serializable so it can be the n8n item payload between nodes.
    Each step reads what it needs and writes its result back.
    """

    question: str
    session_id: str | None = None
    chunks: list[Chunk] = Field(default_factory=list)
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    answer: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResult(BaseModel):
    """Final output of a pipeline.query() call — everything the caller needs."""

    answer: str
    contexts: list[RetrievalResult]
    trace: list[StepTrace]
    metrics: dict[str, Any] = Field(default_factory=dict)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)


# ── Indexing ──────────────────────────────────────────────────────────────────

class IndexStats(BaseModel):
    """Summary returned by pipeline.index()."""

    num_documents: int
    num_chunks: int
    duration_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)
