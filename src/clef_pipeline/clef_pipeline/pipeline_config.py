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
    fusion_top_k: int = 30
    sparse_cache_top_k: int = 2000
    final_top_k: int = 5

    def enabled_retrievers(self) -> list[RetrieverConfig]:
        """Return retriever configs that are enabled."""
        return [retriever for retriever in self.retrievers if retriever.enabled]


def build_pipeline_config(profile: str = "demo") -> PipelineConfig:
    """Build a predefined pipeline configuration profile.

    Args:
        profile: Profile name (``demo``, ``evaluation``, or ``retrieval-only``).

    Returns:
        Fully populated pipeline configuration.

    Raises:
        ValueError: If ``profile`` is unknown.
    """
    if profile == "demo":
        return PipelineConfig(
            retrievers=[
                RetrieverConfig(name="harrier-270m"),
                RetrieverConfig(name="sparse"),
            ],
            reranker=RerankerConfig(name="nemotron", enabled=True),
            use_fusion=True,
            fusion_top_k=30,
            sparse_cache_top_k=2000,
            final_top_k=5,
        )

    if profile == "evaluation":
        return PipelineConfig(
            retrievers=[
                RetrieverConfig(name="harrier-27b"),
                RetrieverConfig(name="sparse"),
            ],
            reranker=RerankerConfig(name="nemotron", enabled=True),
            use_fusion=True,
            fusion_top_k=30,
            sparse_cache_top_k=2000,
            final_top_k=5,
        )

    if profile == "retrieval-only":
        return PipelineConfig(
            retrievers=[
                RetrieverConfig(name="harrier-270m"),
                RetrieverConfig(name="sparse"),
            ],
            reranker=RerankerConfig(name=None, enabled=False),
            use_fusion=True,
            fusion_top_k=30,
            sparse_cache_top_k=2000,
            final_top_k=5,
        )

    raise ValueError(f"Unknown pipeline profile: {profile}")
