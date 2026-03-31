"""Configuration for the retrieval pipeline."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Supported semantic reranker backends (D-01)
RerankerBackend = Literal["jina_v2", "bge_v2_m3"]


class RetrievalConfig(BaseModel):
    embedding_model: str = Field(default="models/gemini-embedding-2-preview")
    rerank_model: str = Field(default="gemini-3.1-flash-lite-preview")
    query_model: str = Field(default="gemini-3.1-flash-lite-preview")
    top_k: int = Field(default=200, ge=10)
    top_n: int = Field(default=5, ge=1, le=20)
    cache_dir: str = Field(default=".cache/clef_retrieval")
    embed_batch_size: int = Field(default=100, ge=1, le=100)
    embed_min_interval_seconds: float = Field(default=1.1, gt=0.0)
    embed_max_retries: int = Field(default=5, ge=0, le=20)
    embed_backoff_base_seconds: float = Field(default=1.5, gt=0.0)
    query_batch_size: int = Field(default=32, ge=1, le=100)
    extraction_gate_min_title_mentions: int = Field(default=1, ge=0, le=5)
    extraction_gate_min_authors: int = Field(default=1, ge=0, le=5)
    extraction_gate_min_filled_key_fields: int = Field(default=2, ge=1, le=5)
    weight_title_author: float = Field(default=5.0, gt=0.0)
    weight_method_finding: float = Field(default=3.0, gt=0.0)
    weight_keywords: float = Field(default=1.0, gt=0.0)
    language_normalization_mode: Literal["strict"] = "strict"
    subset_seed: int = Field(default=42)
    subset_per_language_limit: int = Field(default=100, ge=1)

    # Semantic reranker configuration (Phase 2)
    # D-02: Default to jina_v2 backend
    reranker_backend: RerankerBackend = Field(default="jina_v2")
    # D-03: Rerank top 50 dense-retrieval candidates per query
    rerank_top_k: int = Field(default=50, ge=1)
    # Tie-break epsilon for semantic score equivalence
    semantic_tie_epsilon: float = Field(default=0.01, gt=0.0)
    # Backend-specific model identifiers
    jina_reranker_model: str = Field(
        default="jinaai/jina-reranker-v2-base-multilingual"
    )
    bge_reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")

    @model_validator(mode="after")
    def validate_policy_ordering(self) -> "RetrievalConfig":
        if not (
            self.weight_title_author > self.weight_method_finding > self.weight_keywords
        ):
            raise ValueError(
                "weight ordering must be title+author > method/finding > keywords"
            )
        return self
