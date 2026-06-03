"""Shared pytest fixtures.

Uses deterministic fakes — no real APIs, no network calls, no file I/O in unit tests.
"""

from __future__ import annotations

import pytest

from rag_lab.config.schema import EmbeddingsConfig, LLMConfig, RagLabConfig, VectorDBConfig
from rag_lab.core.types import Chunk, Document, RetrievalResult


# ── Fake providers ─────────────────────────────────────────────────────────────

class FakeLLM:
    """Returns a deterministic string for every prompt."""

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        return "fake answer"

    def stream(self, messages: list[dict[str, str]], **kwargs: object):  # type: ignore[return]
        yield "fake "
        yield "answer"


class FakeEmbedder:
    """Returns zero vectors of length 4."""

    @property
    def dimension(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 1.0]] * len(texts)

    def embed_one(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0, 1.0]


class FakeVectorStore:
    """In-memory store for testing."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[list[float], dict[str, object]]] = {}

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, object]],
    ) -> None:
        for id_, vec, payload in zip(ids, vectors, payloads):
            self._data[id_] = (vec, payload)

    def search(
        self,
        vector: list[float],
        top_k: int,
        filter: dict[str, object] | None = None,
    ) -> list[RetrievalResult]:
        results = []
        for rank, (id_, (_, payload)) in enumerate(list(self._data.items())[:top_k]):
            chunk = Chunk(
                id=id_,
                doc_id=str(payload.get("doc_id", "")),
                text=str(payload.get("text", "")),
                metadata={k: v for k, v in payload.items() if k not in ("doc_id", "text")},
            )
            results.append(RetrievalResult(chunk=chunk, score=1.0, rank=rank))
        return results

    def delete(self, ids: list[str]) -> None:
        for id_ in ids:
            self._data.pop(id_, None)

    def count(self) -> int:
        return len(self._data)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(id="doc1", text="Paris is the capital of France.", metadata={"source": "wiki"}),
        Document(id="doc2", text="Berlin is the capital of Germany.", metadata={"source": "wiki"}),
        Document(id="doc3", text="Tokyo is the capital of Japan.", metadata={"source": "wiki"}),
    ]


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(id="c1", doc_id="doc1", text="Paris is the capital of France."),
        Chunk(id="c2", doc_id="doc2", text="Berlin is the capital of Germany."),
    ]


@pytest.fixture
def minimal_config() -> RagLabConfig:
    """A RagLabConfig with no real provider keys required."""
    return RagLabConfig(
        llm=LLMConfig(provider="fake", profile="fake"),
        embeddings=EmbeddingsConfig(provider="fake", profile="fake", dimension=4),
        vectordb=VectorDBConfig(provider="fake", profile="fake"),
    )
