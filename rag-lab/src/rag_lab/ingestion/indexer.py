"""Indexer — the bridge between raw documents and a searchable corpus.

Takes documents, runs them through the chunker, embeds every chunk,
upserts into the vector store, and builds a BM25 sparse index alongside.

After indexing, both DenseRetriever (vector similarity) and BM25Retriever
(keyword matching) can search the same corpus. Hybrid RAG uses both.

n8n node mapping: three sequential nodes — "Chunk Text", "Embed", "Upsert to Store"
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import structlog

from rag_lab.core.errors import IndexError, ProviderError
from rag_lab.core.interfaces import EmbeddingProvider, VectorStore
from rag_lab.core.types import Chunk, Document, IndexStats
from rag_lab.ingestion.chunkers import Chunker
from rag_lab.ingestion.loaders import load_documents

log = structlog.get_logger(__name__)


class Indexer:
    """Chunk → embed → store in Chroma + build BM25 index.

    Handles incremental indexing: new documents are merged with the existing
    BM25 index (deduped by chunk ID), so you can call index() multiple times
    without losing previously indexed content.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        chunker: Chunker,
        bm25_index_path: str = ".bm25_index.pkl",
        batch_size: int = 64,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chunker = chunker
        self.bm25_index_path = Path(bm25_index_path)
        self.batch_size = batch_size

    # ── Public API ────────────────────────────────────────────────────────────

    def index(self, documents: list[Document]) -> IndexStats:
        """Chunk, embed, and store a batch of documents.

        Also rebuilds the BM25 index, merging these chunks with any
        previously indexed chunks.
        """
        t0 = time.perf_counter()

        # 1. Chunk all documents.
        chunks: list[Chunk] = []
        for doc in documents:
            doc_chunks = self.chunker.chunk(doc)
            chunks.extend(doc_chunks)
            log.debug("chunked", doc_id=doc.id, n_chunks=len(doc_chunks))

        if not chunks:
            log.warning("no_chunks_produced", n_docs=len(documents))
            return IndexStats(num_documents=len(documents), num_chunks=0, duration_ms=0.0)

        # 2. Embed + upsert to vector store (in batches to cap memory).
        self._embed_and_upsert(chunks)

        # 3. Rebuild BM25 index (merging with existing).
        self._rebuild_bm25(chunks)

        duration_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "indexed",
            n_docs=len(documents),
            n_chunks=len(chunks),
            duration_ms=round(duration_ms, 1),
        )
        return IndexStats(
            num_documents=len(documents),
            num_chunks=len(chunks),
            duration_ms=duration_ms,
        )

    def index_file(self, path: str | Path) -> IndexStats:
        """Convenience wrapper: load + index a single file or directory."""
        documents = load_documents(path)
        return self.index(documents)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _embed_and_upsert(self, chunks: list[Chunk]) -> None:
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [c.text for c in batch]
            try:
                vectors = self.embedder.embed(texts)
            except Exception as exc:
                raise ProviderError("embedder", f"Embedding failed: {exc}") from exc

            ids = [c.id for c in batch]
            payloads: list[dict[str, Any]] = [
                {"text": c.text, "doc_id": c.doc_id, **c.metadata} for c in batch
            ]
            self.store.upsert(ids, vectors, payloads)
            log.debug("upserted_batch", start=i, size=len(batch))

    def _rebuild_bm25(self, new_chunks: list[Chunk]) -> None:
        """Merge new_chunks into the BM25 index (dedup by ID, then rebuild)."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 not installed. Run: pip install rank-bm25"
            ) from exc

        # Load previously indexed chunks.
        existing: dict[str, Chunk] = {}
        if self.bm25_index_path.exists():
            try:
                with self.bm25_index_path.open("rb") as f:
                    saved = pickle.load(f)
                existing = {c.id: c for c in saved.get("chunks", [])}
                log.debug("bm25_loaded_existing", n=len(existing))
            except Exception as exc:
                log.warning("bm25_load_failed_rebuilding", error=str(exc))

        # Merge: new chunks override any existing chunk with the same ID.
        for chunk in new_chunks:
            existing[chunk.id] = chunk
        all_chunks = list(existing.values())

        tokenized = [c.text.lower().split() for c in all_chunks]
        bm25 = BM25Okapi(tokenized)

        with self.bm25_index_path.open("wb") as f:
            pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)

        log.debug("bm25_rebuilt", total_chunks=len(all_chunks))
