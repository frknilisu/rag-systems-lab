"""Pydantic models for the full RAG-Lab configuration tree."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    profile: str = "openrouter_free"
    provider: str = "litellm"
    model: str = ""
    api_key_env: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024


class EmbeddingsConfig(BaseModel):
    profile: str = "local_st"
    provider: str = "sentence_transformers"
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384


class VectorDBConfig(BaseModel):
    profile: str = "chroma"
    provider: str = "chroma"
    path: str = ".chroma"
    collection: str = "rag_lab"
    # Qdrant-specific (ignored by Chroma)
    url: str = "http://localhost:6333"
    api_key_env: str | None = None


class RerankerConfig(BaseModel):
    enabled: bool = False
    profile: str = "cross_encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class IngestionConfig(BaseModel):
    chunker: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64


class HybridConfig(BaseModel):
    enabled: bool = False
    fusion: str = "rrf"
    sparse_weight: float = 0.5


class RetrievalConfig(BaseModel):
    top_k: int = 5
    hybrid: HybridConfig = Field(default_factory=HybridConfig)


# ── Per-architecture sub-configs ──────────────────────────────────────────────

class RecursiveConfig(BaseModel):
    max_hops: int = 3


class SelfRagConfig(BaseModel):
    relevance_threshold: float = 0.6
    max_retrieval_rounds: int = 2


class HyDEConfig(BaseModel):
    n_hypotheticals: int = 1


class QueryTransformConfig(BaseModel):
    mode: str = "decompose"


class MemoryRagConfig(BaseModel):
    store: str = "vector"
    max_memory_items: int = 20


class StreamingConfig(BaseModel):
    window_seconds: int = 3600
    refresh_seconds: int = 60


class ArchitecturesConfig(BaseModel):
    recursive: RecursiveConfig = Field(default_factory=RecursiveConfig)
    self_rag: SelfRagConfig = Field(default_factory=SelfRagConfig)
    hyde: HyDEConfig = Field(default_factory=HyDEConfig)
    query_transform: QueryTransformConfig = Field(default_factory=QueryTransformConfig)
    memory_rag: MemoryRagConfig = Field(default_factory=MemoryRagConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)


# ── Top-level ─────────────────────────────────────────────────────────────────

class RagLabConfig(BaseModel):
    log_level: str = "INFO"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vectordb: VectorDBConfig = Field(default_factory=VectorDBConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    architectures: ArchitecturesConfig = Field(default_factory=ArchitecturesConfig)
