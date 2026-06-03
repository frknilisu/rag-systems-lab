"""Standard RAG pipeline — the foundational architecture.

Implements the simplest complete RAG loop:
  index  : chunk → embed → store in Chroma + BM25
  query  : embed question → dense search → grounded LLM generation

This is architecture #1 and the baseline against which all others are compared.
Every other architecture in RAG-Lab either extends or replaces part of this loop.

See docs/architectures/standard.md for the full teaching chapter.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from rag_lab.architectures.base import RAGPipeline
from rag_lab.architectures.standard.steps import generate_step, retrieve_step
from rag_lab.config.schema import RagLabConfig
from rag_lab.core.pipeline import Pipeline, PipelineDeps
from rag_lab.core.registry import (
    build_embedding,
    build_llm,
    build_vectorstore,
    register_architecture,
)
from rag_lab.core.types import Document, IndexStats, RAGResult, RAGState
from rag_lab.ingestion.chunkers import build_chunker
from rag_lab.ingestion.indexer import Indexer

logger = structlog.get_logger(__name__)


@register_architecture("standard")
class StandardRAGPipeline(RAGPipeline):
    """Retrieve → Generate: the simplest complete RAG architecture.

    All other architectures in this repo are either extensions or alternatives
    to this two-step loop. It is the baseline for every evaluation comparison.

    Args:
        config: Full RagLabConfig. Provider selection, chunk sizes, top-k,
                and LLM parameters all come from here — no magic numbers.
        deps:   Optional pre-built PipelineDeps. When provided (e.g. in tests),
                no real providers are constructed from config. When None, all
                providers are built via the registry factories.
    """

    name = "standard"

    def __init__(
        self,
        config: RagLabConfig,
        *,
        deps: PipelineDeps | None = None,
    ) -> None:
        self._cfg = config

        if deps is not None:
            self._deps = deps
            embedder = deps.embedder
            store = deps.store
        else:
            llm = build_llm(config.llm)
            embedder = build_embedding(config.embeddings)
            store = build_vectorstore(config.vectordb)
            self._deps = PipelineDeps(
                llm=llm,
                embedder=embedder,
                store=store,
                extra={"top_k": config.retrieval.top_k},
            )

        self._indexer = Indexer(
            embedder=embedder,
            store=store,
            chunker=build_chunker(config.ingestion),
            bm25_index_path=config.ingestion.bm25_index_path,
        )

    # ── RAGPipeline interface ─────────────────────────────────────────────────

    def index(self, documents: list[Document]) -> IndexStats:
        """Chunk, embed, and store documents. Safe to call incrementally.

        Steps: Chunk → Embed (batched) → Upsert to Chroma → Rebuild BM25.
        """
        t0 = time.perf_counter()
        stats = self._indexer.index(documents)
        logger.info(
            "standard_rag.indexed",
            n_docs=stats.num_documents,
            n_chunks=stats.num_chunks,
            duration_ms=round(stats.duration_ms, 1),
        )
        return stats

    def query(
        self,
        question: str,
        *,
        session_id: str | None = None,
    ) -> RAGResult:
        """Retrieve relevant chunks and generate a grounded answer.

        Pipeline steps
        --------------
        1. retrieve — embed question → cosine search → top-k RetrievalResults
        2. generate — build RAG prompt → LLM completion

        The config snapshot (provider names, top-k, model, etc.) is stored in
        RAGResult so every result is self-documenting and reproducible.
        """
        pipeline = Pipeline(
            steps=[
                ("retrieve", retrieve_step),
                ("generate", generate_step),
            ],
            deps=self._deps,
        )
        state = RAGState(question=question, session_id=session_id)

        try:
            state, traces = pipeline.run(state)
        except Exception as exc:
            logger.error("standard_rag.query_failed", question=question[:80], error=str(exc))
            raise

        logger.info(
            "standard_rag.query_done",
            question=question[:80],
            n_contexts=len(state.retrieval_results),
            answer_chars=len(state.answer),
        )

        return RAGResult(
            answer=state.answer,
            contexts=state.retrieval_results,
            trace=traces,
            config_snapshot=self._cfg.model_dump(),
        )

    # ── Convenience ───────────────────────────────────────────────────────────

    def index_file(self, path: str) -> IndexStats:
        """Convenience wrapper: load + index a file or directory."""
        return self._indexer.index_file(path)

    @property
    def store_count(self) -> int:
        """Number of chunks currently in the vector store."""
        store: Any = self._deps.store
        return store.count() if hasattr(store, "count") else 0
