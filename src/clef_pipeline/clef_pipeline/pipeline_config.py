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
    fusion_top_k: int = 100
    sparse_cache_top_k: int = 2000
    final_top_k: int = 5
    # Per-retriever weights applied by the fuser (aligned with enabled retrievers).
    # Dense-heavy default matches the empirical finding that dense dominates sparse
    # on this task (Dense MRR@5 0.69 vs Sparse 0.53).
    fusion_weights: tuple[float, ...] | None = None

    def enabled_retrievers(self) -> list[RetrieverConfig]:
        """Return retriever configs that are enabled."""
        return [retriever for retriever in self.retrievers if retriever.enabled]


_ALLOWED_FUSION_METHODS = {"rrf", "random_forest"}


def build_pipeline_config(
    profile: str = "demo",
    fusion_method: str = "rrf",
    fusion_weights: tuple[float, ...] | None = (0.8, 0.2),
    fusion_top_k: int = 100,
    extra_retrievers: tuple[RetrieverConfig, ...] = (),
    reranker_name: str | None = None,
    reranker_params: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Build a predefined pipeline configuration profile.

    Args:
        profile: Profile name (``demo``, ``evaluation``, or ``retrieval-only``).
        fusion_method: ``rrf`` or ``random_forest``.
        fusion_weights: Optional per-retriever weight vector aligned with the
            enabled retrievers (dense first, sparse second in all presets).
            Defaults to ``(0.8, 0.2)`` — dense-heavy because dense dominates
            sparse on this task.
        fusion_top_k: Candidate pool handed to the reranker. Raised from 30
            to 100 so the reranker sees more of the gold docs.
        reranker_name: Optional override for the profile's default reranker
            registry name (e.g. ``"qwen3-reranker-8b"``). When ``None``, each
            profile keeps its own default. Pass an empty string to explicitly
            disable reranking.
        reranker_params: Optional constructor kwargs forwarded to the
            reranker factory (e.g. ``{"model_name": "Qwen/Qwen3-Reranker-8B"}``).

    Returns:
        Fully populated pipeline configuration.

    Raises:
        ValueError: If ``profile`` or ``fusion_method`` is unknown.
    """
    normalized_fusion_method = fusion_method.strip().lower()
    if normalized_fusion_method not in _ALLOWED_FUSION_METHODS:
        raise ValueError(
            f"Unknown fusion method: {fusion_method}. "
            f"Use one of {sorted(_ALLOWED_FUSION_METHODS)}."
        )

    extras = list(extra_retrievers)

    def _reranker(default: RerankerConfig) -> RerankerConfig:
        """Apply caller overrides on top of the profile's default reranker."""
        if reranker_name is None and reranker_params is None:
            return default
        # Empty-string name disables reranking regardless of other inputs.
        if reranker_name == "":
            return RerankerConfig(name=None, enabled=False)
        effective_name = reranker_name if reranker_name is not None else default.name
        effective_params = (
            dict(reranker_params) if reranker_params is not None else dict(default.params)
        )
        enabled = effective_name is not None
        return RerankerConfig(
            name=effective_name, enabled=enabled, params=effective_params
        )

    if profile == "demo":
        retrievers = [
            RetrieverConfig(name="harrier-270m"),
            RetrieverConfig(name="sparse"),
            *extras,
        ]
        return PipelineConfig(
            retrievers=retrievers,
            reranker=_reranker(RerankerConfig(name="nemotron", enabled=True)),
            use_fusion=True,
            fusion_method=normalized_fusion_method,
            fusion_top_k=fusion_top_k,
            sparse_cache_top_k=2000,
            final_top_k=5,
            fusion_weights=fusion_weights,
        )

    if profile == "evaluation":
        retrievers = [
            RetrieverConfig(name="harrier-27b"),
            RetrieverConfig(name="sparse"),
            *extras,
        ]
        return PipelineConfig(
            retrievers=retrievers,
            reranker=_reranker(RerankerConfig(name="nemotron", enabled=True)),
            use_fusion=True,
            fusion_method=normalized_fusion_method,
            fusion_top_k=fusion_top_k,
            sparse_cache_top_k=2000,
            final_top_k=5,
            fusion_weights=fusion_weights,
        )

    if profile == "retrieval-only":
        retrievers = [
            RetrieverConfig(name="harrier-270m"),
            RetrieverConfig(name="sparse"),
            *extras,
        ]
        return PipelineConfig(
            retrievers=retrievers,
            reranker=_reranker(RerankerConfig(name=None, enabled=False)),
            use_fusion=True,
            fusion_method=normalized_fusion_method,
            fusion_top_k=fusion_top_k,
            sparse_cache_top_k=2000,
            final_top_k=5,
            fusion_weights=fusion_weights,
        )

    raise ValueError(f"Unknown pipeline profile: {profile}")
