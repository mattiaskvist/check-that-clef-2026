"""Configuration for the retrieval pipeline."""

from pydantic import BaseModel, Field


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
