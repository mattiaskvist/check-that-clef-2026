"""Reranker scaffolding models and helpers."""

from pydantic import BaseModel, Field


class RerankResult(BaseModel):
    ranked_pubkeys: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    rationale_short: list[str] = Field(default_factory=list)


def clip_top5(ranked_pubkeys: list[str]) -> list[str]:
    return ranked_pubkeys[:5]
