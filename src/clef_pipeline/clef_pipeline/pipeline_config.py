"""Configuration models and presets for retrieval pipeline assembly."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrieverConfig:
    """Configuration for one retriever implementation."""

    name: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankerConfig:
    """Configuration for the optional reranker stage."""

    name: str | None = None
    enabled: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level settings used to build and run the retrieval pipeline."""

    retrievers: list[RetrieverConfig]
    reranker: RerankerConfig
    use_fusion: bool = True
    fusion_method: str = "rrf"
    fusion_top_k: int = 30
    sparse_cache_top_k: int = 2000
    final_top_k: int = 5
    hf_fusion_repo_id: str | None = None
    hf_token: str | None = None

    def enabled_retrievers(self) -> list[RetrieverConfig]:
        """Return retriever configs that are enabled."""
        return [retriever for retriever in self.retrievers if retriever.enabled]


def build_pipeline_config(
    profile: str = "demo",
    fusion_method: str = "rrf",
    fusion_top_k: int = 30,
    hf_fusion_repo_id: str | None = None,
    hf_token: str | None = None,
    dense_model: str = "harrier-27b",
    disable_sparse: bool = False,
    reranker_model: str = "nemotron",
    disable_reranker: bool = False,
    sparse_k1: float = 2.5,
    sparse_b: float = 0.85,
    sparse_use_bigrams: bool = True,
    sparse_use_translation: bool = True,
) -> PipelineConfig:
    """Build a predefined pipeline configuration profile.

    Args:
        profile: Profile name (``demo``, ``evaluation``, or ``retrieval-only``).

    Returns:
        Fully populated pipeline configuration.

    Raises:
        ValueError: If ``profile`` is unknown.
    """
    normalized_fusion_method = fusion_method.strip().lower()
    if normalized_fusion_method not in {"rrf", "random_forest"}:
        raise ValueError(
            f"Unknown fusion method: {fusion_method}. Use 'rrf' or 'random_forest'."
        )

    if profile == "custom":
        retrievers = []
        if dense_model:
            retrievers.append(RetrieverConfig(name=dense_model))
        if not disable_sparse:
            retrievers.append(
                RetrieverConfig(
                    name="sparse",
                    params={
                        "k1": sparse_k1,
                        "b": sparse_b,
                        "use_bigrams": sparse_use_bigrams,
                        "use_translation": sparse_use_translation,
                    },
                )
            )

        return PipelineConfig(
            retrievers=retrievers,
            reranker=RerankerConfig(name=reranker_model, enabled=not disable_reranker),
            use_fusion=True,
            fusion_method=normalized_fusion_method,
            fusion_top_k=fusion_top_k,
            sparse_cache_top_k=2000,
            final_top_k=5,
            hf_fusion_repo_id=hf_fusion_repo_id,
            hf_token=hf_token,
        )

    if profile == "demo":
        return PipelineConfig(
            retrievers=[
                RetrieverConfig(name="harrier-270m"),
                RetrieverConfig(name="sparse"),
            ],
            reranker=RerankerConfig(name="nemotron", enabled=True),
            use_fusion=True,
            fusion_method=normalized_fusion_method,
            fusion_top_k=fusion_top_k,
            sparse_cache_top_k=2000,
            final_top_k=5,
            hf_fusion_repo_id=hf_fusion_repo_id,
            hf_token=hf_token,
        )

    if profile == "evaluation":
        return PipelineConfig(
            retrievers=[
                RetrieverConfig(name="harrier-27b"),
                RetrieverConfig(name="sparse"),
            ],
            reranker=RerankerConfig(name="nemotron", enabled=True),
            use_fusion=True,
            fusion_method=normalized_fusion_method,
            fusion_top_k=fusion_top_k,
            sparse_cache_top_k=2000,
            final_top_k=5,
            hf_fusion_repo_id=hf_fusion_repo_id,
            hf_token=hf_token,
        )

    if profile == "retrieval-only":
        return PipelineConfig(
            retrievers=[
                RetrieverConfig(name="harrier-270m"),
                RetrieverConfig(name="sparse"),
            ],
            reranker=RerankerConfig(name=None, enabled=False),
            use_fusion=True,
            fusion_method=normalized_fusion_method,
            fusion_top_k=fusion_top_k,
            sparse_cache_top_k=2000,
            final_top_k=5,
            hf_fusion_repo_id=hf_fusion_repo_id,
            hf_token=hf_token,
        )

    raise ValueError(f"Unknown pipeline profile: {profile}")
