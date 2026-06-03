"""Provider registry — maps string names to implementation classes.

Usage (provider side):
    @register_llm("litellm")
    class LiteLLMProvider: ...

Usage (consumer side):
    llm = build_llm(config.llm)
    embedder = build_embedding(config.embeddings)
    store = build_vectorstore(config.vectordb)

Providers self-register by being imported; each providers/__init__.py
imports its implementations to trigger registration.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from rag_lab.config.schema import EmbeddingsConfig, LLMConfig, VectorDBConfig
from rag_lab.core.errors import ConfigError

T = TypeVar("T")

_LLM_REGISTRY: dict[str, type] = {}
_EMBEDDING_REGISTRY: dict[str, type] = {}
_VECTORSTORE_REGISTRY: dict[str, type] = {}
_ARCHITECTURE_REGISTRY: dict[str, type] = {}


# ── Decorators ────────────────────────────────────────────────────────────────

def register_llm(name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        _LLM_REGISTRY[name] = cls
        return cls
    return decorator


def register_embedding(name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        _EMBEDDING_REGISTRY[name] = cls
        return cls
    return decorator


def register_vectorstore(name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        _VECTORSTORE_REGISTRY[name] = cls
        return cls
    return decorator


def register_architecture(name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        _ARCHITECTURE_REGISTRY[name] = cls
        return cls
    return decorator


# ── Factories ─────────────────────────────────────────────────────────────────

def build_llm(config: LLMConfig) -> Any:
    """Instantiate the LLM provider named in config.provider."""
    _ensure_providers_loaded()
    cls = _LLM_REGISTRY.get(config.provider)
    if cls is None:
        raise ConfigError(
            f"Unknown LLM provider '{config.provider}'. "
            f"Available: {sorted(_LLM_REGISTRY)}"
        )
    return cls(config)


def build_embedding(config: EmbeddingsConfig) -> Any:
    """Instantiate the embedding provider named in config.provider."""
    _ensure_providers_loaded()
    cls = _EMBEDDING_REGISTRY.get(config.provider)
    if cls is None:
        raise ConfigError(
            f"Unknown embedding provider '{config.provider}'. "
            f"Available: {sorted(_EMBEDDING_REGISTRY)}"
        )
    return cls(config)


def build_vectorstore(config: VectorDBConfig) -> Any:
    """Instantiate the vector store named in config.provider."""
    _ensure_providers_loaded()
    cls = _VECTORSTORE_REGISTRY.get(config.provider)
    if cls is None:
        raise ConfigError(
            f"Unknown vector store '{config.provider}'. "
            f"Available: {sorted(_VECTORSTORE_REGISTRY)}"
        )
    return cls(config)


def build_architecture(name: str, **kwargs: Any) -> Any:
    cls = _ARCHITECTURE_REGISTRY.get(name)
    if cls is None:
        raise ConfigError(
            f"Unknown architecture '{name}'. "
            f"Available: {sorted(_ARCHITECTURE_REGISTRY)}"
        )
    return cls(**kwargs)


def list_providers() -> dict[str, list[str]]:
    _ensure_providers_loaded()
    return {
        "llm": sorted(_LLM_REGISTRY),
        "embeddings": sorted(_EMBEDDING_REGISTRY),
        "vectorstores": sorted(_VECTORSTORE_REGISTRY),
        "architectures": sorted(_ARCHITECTURE_REGISTRY),
    }


# ── Lazy provider import ───────────────────────────────────────────────────────

_providers_loaded = False


def _ensure_providers_loaded() -> None:
    global _providers_loaded
    if _providers_loaded:
        return
    # Import provider packages; their __init__.py triggers @register_* calls.
    import rag_lab.providers  # noqa: F401
    _providers_loaded = True
