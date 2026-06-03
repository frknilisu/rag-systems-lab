"""Sentence-Transformers embedding provider.

Runs fully locally — no API key, no cost, no network after the first download.
The model (~90 MB for all-MiniLM-L6-v2) is cached by HuggingFace Hub.

n8n node mapping: "Embeddings" Function node (or Embeddings node in n8n AI)
"""

from __future__ import annotations

import structlog

from rag_lab.config.schema import EmbeddingsConfig
from rag_lab.core.errors import ProviderError
from rag_lab.core.registry import register_embedding

log = structlog.get_logger(__name__)


@register_embedding("sentence_transformers")
class SentenceTransformersProvider:
    """Wraps sentence_transformers.SentenceTransformer for dense embeddings."""

    def __init__(self, config: EmbeddingsConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ProviderError(
                "sentence_transformers",
                "sentence-transformers not installed. Run: pip install rag-lab[embeddings]",
            ) from exc

        self._model = SentenceTransformer(config.model)
        self._dimension = config.dimension
        log.debug("sentence_transformers_ready", model=config.model, dim=config.dimension)

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._model.encode(texts, convert_to_numpy=True)
            return vectors.tolist()  # type: ignore[union-attr]
        except Exception as exc:
            raise ProviderError("sentence_transformers", str(exc)) from exc

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension
