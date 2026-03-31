"""Reranker scaffolding models and helpers.

Provides semantic reranker backend abstraction and scoring/tie-break primitives
for Phase 2 semantic-primary reranking (D-01, D-02, D-05, D-06).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .config import RetrievalConfig


class RerankResult(BaseModel):
    ranked_pubkeys: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    rationale_short: list[str] = Field(default_factory=list)


def clip_top5(ranked_pubkeys: list[str]) -> list[str]:
    return ranked_pubkeys[:5]


@runtime_checkable
class SemanticReranker(Protocol):
    """Protocol for semantic reranker backends.

    Implementations must provide a score_candidates method that returns
    (pubkey, score) tuples for each candidate.
    """

    def score_candidates(
        self,
        query: str,
        candidates: list[str],
        metadata_by_pubkey: dict[str, dict],
    ) -> list[tuple[str, float]]:
        """Score query-document pairs for reranking.

        Args:
            query: Query text to match against candidates
            candidates: List of pubkeys to score
            metadata_by_pubkey: Dict mapping pubkey to document metadata
                (must have 'title' and 'abstract' keys)

        Returns:
            List of (pubkey, score) tuples in original candidate order
        """
        ...


class BaseRerankerAdapter(ABC):
    """Base class for reranker backend adapters."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None

    @abstractmethod
    def _load_model(self):
        """Load the model. Called lazily on first use."""
        pass

    @abstractmethod
    def _compute_scores(
        self, pairs: list[tuple[str, str]]
    ) -> list[float]:
        """Compute relevance scores for query-document pairs."""
        pass

    def score_candidates(
        self,
        query: str,
        candidates: list[str],
        metadata_by_pubkey: dict[str, dict],
    ) -> list[tuple[str, float]]:
        """Score candidates using cross-encoder model."""
        if not candidates:
            return []

        # Build query-document pairs
        pairs = []
        for pubkey in candidates:
            meta = metadata_by_pubkey.get(pubkey, {})
            doc_text = f"{meta.get('title', '')} {meta.get('abstract', '')}".strip()
            pairs.append((query, doc_text))

        # Compute scores
        scores = self._compute_scores(pairs)

        return [(pubkey, score) for pubkey, score in zip(candidates, scores)]


class JinaRerankerAdapter(BaseRerankerAdapter):
    """Adapter for Jina Reranker v2 backend.

    Uses AutoModelForSequenceClassification with trust_remote_code=True
    per Jina model card recommendations.
    """

    def _load_model(self):
        try:
            from transformers import AutoModelForSequenceClassification

            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                trust_remote_code=True,
            )
        except ImportError as e:
            raise RuntimeError(
                f"Jina reranker backend requires 'transformers' package. "
                f"Install with: uv add transformers torch"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Jina reranker model '{self.model_id}': {e}"
            ) from e

    def _compute_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._model is None:
            self._load_model()

        if not pairs:
            return []

        try:
            # Jina models provide compute_score method
            scores = self._model.compute_score(
                [[q, d] for q, d in pairs],
                max_length=1024,
            )
            # Handle both single score and list return
            if isinstance(scores, (int, float)):
                return [float(scores)]
            return [float(s) for s in scores]
        except Exception as e:
            raise RuntimeError(f"Jina reranker scoring failed: {e}") from e


class BGERerankerAdapter(BaseRerankerAdapter):
    """Adapter for BGE Reranker v2-m3 backend.

    Supports both FlagEmbedding (preferred) and sentence-transformers CrossEncoder.
    """

    def _load_model(self):
        # Try FlagEmbedding first (recommended for BGE models)
        try:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(self.model_id, use_fp16=True)
            self._use_flag = True
            return
        except ImportError:
            pass

        # Fall back to sentence-transformers CrossEncoder
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id)
            self._use_flag = False
            return
        except ImportError:
            pass

        # Neither available
        raise RuntimeError(
            f"BGE reranker backend requires either 'FlagEmbedding' or "
            f"'sentence-transformers' package. "
            f"Install with: uv add FlagEmbedding or uv add sentence-transformers"
        )

    def _compute_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._model is None:
            self._load_model()

        if not pairs:
            return []

        try:
            formatted_pairs = [[q, d] for q, d in pairs]

            if getattr(self, "_use_flag", False):
                scores = self._model.compute_score(formatted_pairs)
            else:
                # sentence-transformers CrossEncoder
                scores = self._model.predict(formatted_pairs)

            # Handle both single score and list return
            if isinstance(scores, (int, float)):
                return [float(scores)]
            return [float(s) for s in scores]
        except Exception as e:
            raise RuntimeError(f"BGE reranker scoring failed: {e}") from e


class MockReranker:
    """Mock reranker for unit testing (no model loading required).

    Returns predictable scores based on document position for testing
    deterministic behavior.
    """

    def score_candidates(
        self,
        query: str,
        candidates: list[str],
        metadata_by_pubkey: dict[str, dict],
    ) -> list[tuple[str, float]]:
        """Return mock scores descending by position."""
        return [(pubkey, 1.0 - (i * 0.1)) for i, pubkey in enumerate(candidates)]


def get_reranker(config: RetrievalConfig) -> SemanticReranker:
    """Factory function to get reranker adapter based on config.

    Args:
        config: RetrievalConfig with reranker_backend field

    Returns:
        SemanticReranker adapter instance for the configured backend
    """
    backend = config.reranker_backend

    if backend == "jina_v2":
        return JinaRerankerAdapter(config.jina_reranker_model)
    elif backend == "bge_v2_m3":
        return BGERerankerAdapter(config.bge_reranker_model)
    else:
        raise ValueError(f"Unknown reranker backend: {backend}")


def semantic_primary_sort(
    candidates_with_scores: list[tuple[str, float, float, int]],
    epsilon: float,
) -> list[tuple[str, float, float, int]]:
    """Sort candidates with semantic score as primary, weighted as tie-break.

    Implements D-05 (semantic-primary) and D-06 (weighted tie-break only):
    1. Sort by semantic score descending (primary)
    2. When semantic scores are within epsilon, use weighted score (secondary)
    3. When both tie, preserve original index order (deterministic fallback)

    Args:
        candidates_with_scores: List of (pubkey, semantic_score, weighted_score, original_index)
        epsilon: Threshold for semantic score equivalence

    Returns:
        Sorted list in semantic-primary order
    """
    if not candidates_with_scores:
        return []

    def sort_key(item: tuple[str, float, float, int]) -> tuple[float, float, int]:
        pubkey, semantic, weighted, orig_idx = item
        # Discretize semantic score to epsilon buckets for tie detection
        # Higher semantic = better, so negate for descending sort
        semantic_bucket = -round(semantic / epsilon) * epsilon
        # Higher weighted = better, so negate for descending sort
        weighted_neg = -weighted
        # Lower original index = earlier, for stable sort
        return (semantic_bucket, weighted_neg, orig_idx)

    return sorted(candidates_with_scores, key=sort_key)
