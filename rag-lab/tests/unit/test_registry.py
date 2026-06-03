"""Tests for core/registry.py — provider registration and factories."""

from __future__ import annotations

import pytest

from rag_lab.core.errors import ConfigError
from rag_lab.core.registry import (
    _EMBEDDING_REGISTRY,
    _LLM_REGISTRY,
    _VECTORSTORE_REGISTRY,
    build_embedding,
    build_llm,
    build_vectorstore,
    list_providers,
    register_embedding,
    register_llm,
    register_vectorstore,
)
from rag_lab.config.schema import EmbeddingsConfig, LLMConfig, VectorDBConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

class _StubLLM:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg

    def complete(self, messages, **kw):
        return "stub"

    def stream(self, messages, **kw):
        yield "stub"


class _StubEmbedder:
    def __init__(self, cfg: EmbeddingsConfig) -> None:
        self.cfg = cfg

    @property
    def dimension(self) -> int:
        return 4

    def embed(self, texts):
        return [[0.0] * 4] * len(texts)

    def embed_one(self, text):
        return [0.0] * 4


class _StubStore:
    def __init__(self, cfg: VectorDBConfig) -> None:
        self.cfg = cfg

    def upsert(self, ids, vectors, payloads): pass
    def search(self, vector, top_k, filter=None): return []
    def delete(self, ids): pass
    def count(self): return 0


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_register_and_build_llm():
    register_llm("_test_llm")(_StubLLM)
    cfg = LLMConfig(provider="_test_llm", profile="test")
    llm = build_llm(cfg)
    assert isinstance(llm, _StubLLM)
    _LLM_REGISTRY.pop("_test_llm", None)


def test_register_and_build_embedding():
    register_embedding("_test_emb")(_StubEmbedder)
    cfg = EmbeddingsConfig(provider="_test_emb", profile="test", dimension=4)
    emb = build_embedding(cfg)
    assert isinstance(emb, _StubEmbedder)
    _EMBEDDING_REGISTRY.pop("_test_emb", None)


def test_register_and_build_vectorstore():
    register_vectorstore("_test_vs")(_StubStore)
    cfg = VectorDBConfig(provider="_test_vs", profile="test")
    store = build_vectorstore(cfg)
    assert isinstance(store, _StubStore)
    _VECTORSTORE_REGISTRY.pop("_test_vs", None)


def test_build_unknown_llm_raises():
    with pytest.raises(ConfigError, match="Unknown LLM provider"):
        build_llm(LLMConfig(provider="__nonexistent__", profile="x"))


def test_build_unknown_embedding_raises():
    with pytest.raises(ConfigError, match="Unknown embedding provider"):
        build_embedding(EmbeddingsConfig(provider="__nonexistent__", profile="x"))


def test_build_unknown_vectorstore_raises():
    with pytest.raises(ConfigError, match="Unknown vector store"):
        build_vectorstore(VectorDBConfig(provider="__nonexistent__", profile="x"))


def test_list_providers_returns_dict():
    providers = list_providers()
    assert "llm" in providers
    assert "embeddings" in providers
    assert "vectorstores" in providers
    assert isinstance(providers["llm"], list)
