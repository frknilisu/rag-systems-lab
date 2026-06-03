"""Chroma vector store provider.

Chroma is a local on-disk vector database — no Docker, no account, data persists
in the `path` directory. Perfect for development and the tutorial examples.

n8n node mapping: "Vector Store" node (Chroma or Pinecone in n8n AI nodes)
"""

from __future__ import annotations

import structlog

from rag_lab.config.schema import VectorDBConfig
from rag_lab.core.errors import ProviderError
from rag_lab.core.registry import register_vectorstore
from rag_lab.core.types import Chunk, RetrievalResult

log = structlog.get_logger(__name__)


@register_vectorstore("chroma")
class ChromaVectorStore:
    """Wraps chromadb.PersistentClient with the VectorStore interface."""

    def __init__(self, config: VectorDBConfig) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ProviderError(
                "chroma",
                "chromadb not installed. Run: pip install rag-lab[chroma]",
            ) from exc

        client = chromadb.PersistentClient(path=config.path)
        self._collection = client.get_or_create_collection(
            name=config.collection,
            metadata={"hnsw:space": "cosine"},
        )
        log.debug(
            "chroma_ready",
            path=config.path,
            collection=config.collection,
            count=self._collection.count(),
        )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, object]],
    ) -> None:
        try:
            self._collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=[str(p.get("text", "")) for p in payloads],
                metadatas=[{k: v for k, v in p.items() if k != "text"} for p in payloads],
            )
        except Exception as exc:
            raise ProviderError("chroma", str(exc)) from exc

    def search(
        self,
        vector: list[float],
        top_k: int,
        filter: dict[str, object] | None = None,
    ) -> list[RetrievalResult]:
        try:
            results = self._collection.query(
                query_embeddings=[vector],
                n_results=min(top_k, max(self._collection.count(), 1)),
                where=filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise ProviderError("chroma", str(exc)) from exc

        retrieval: list[RetrievalResult] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for rank, (chunk_id, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, dists)
        ):
            meta = meta or {}
            chunk = Chunk(
                id=chunk_id,
                doc_id=str(meta.get("doc_id", "")),
                text=doc,
                metadata=meta,
            )
            # Chroma returns cosine distance (0=identical, 2=opposite);
            # convert to similarity score in [0, 1].
            score = 1.0 - (dist / 2.0)
            retrieval.append(RetrievalResult(chunk=chunk, score=score, rank=rank))

        return retrieval

    def delete(self, ids: list[str]) -> None:
        try:
            self._collection.delete(ids=ids)
        except Exception as exc:
            raise ProviderError("chroma", str(exc)) from exc

    def count(self) -> int:
        return self._collection.count()
