"""Cross-encoder reranker — rescore retrieved chunks with a bi-directional model.

A dense retriever uses a *bi-encoder*: query and document are encoded
separately, then compared with dot-product. That's fast but coarse.

A *cross-encoder* sees query and document together, letting attention flow
freely between them. This gives much better relevance scores, at the cost of
running inference once per (query, chunk) pair.

The typical pattern: retrieve top-50 with a fast bi-encoder, then rerank
with a cross-encoder to get the top-5 you actually send to the LLM.

n8n node mapping: "Rerank" Function node between "Dense Search" and "Generate".
"""

from __future__ import annotations

import structlog

from rag_lab.core.errors import ProviderError
from rag_lab.core.types import RetrievalResult

log = structlog.get_logger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Rerank retrieval results using a sentence-transformers cross-encoder.

    Args:
        model: HuggingFace model ID for the cross-encoder.
               Default is a fast, good-quality MS MARCO model (~85 MB).
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self.model_name = model
        self._model: object | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            log.debug("cross_encoder_loaded", model=self.model_name)
        except ImportError as exc:
            raise ProviderError(
                "cross_encoder",
                "sentence-transformers not installed. Run: pip install rag-lab[rerank]",
            ) from exc

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rescore and re-sort results; optionally truncate to top_k.

        Args:
            query: The original query string.
            results: Initial retrieval results (from dense or hybrid).
            top_k: If set, keep only this many results after reranking.

        Returns:
            Results re-ordered by cross-encoder score (highest first).
        """
        if not results:
            return results

        self._load()
        pairs = [(query, r.chunk.text) for r in results]

        try:
            scores: list[float] = self._model.predict(pairs).tolist()  # type: ignore[union-attr]
        except Exception as exc:
            raise ProviderError("cross_encoder", f"Reranking failed: {exc}") from exc

        reranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        final = [
            RetrievalResult(chunk=r.chunk, score=float(score), rank=rank)
            for rank, (r, score) in enumerate(reranked)
        ]
        log.debug(
            "reranked",
            query=query[:60],
            n_in=len(results),
            n_out=len(final),
            top_score=final[0].score if final else None,
        )
        return final
