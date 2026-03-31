"""Deterministic policy helpers for query-understanding behavior."""

from __future__ import annotations

import random
import re
import unicodedata

from .config import RetrievalConfig
from .schemas import TweetEvidence


def extraction_gate_passes(
    evidence: TweetEvidence | None,
    config: RetrievalConfig,
) -> bool:
    """Return True when extracted evidence passes strict policy thresholds."""
    if evidence is None:
        return False

    title_ok = len(evidence.candidate_title_mentions) >= config.extraction_gate_min_title_mentions
    authors_ok = len(evidence.candidate_authors) >= config.extraction_gate_min_authors
    method_or_finding_ok = bool(evidence.method_terms or evidence.finding_terms)
    keywords_ok = bool(evidence.keywords)
    summary_ok = bool(evidence.claim_summary.strip())

    filled_key_fields = sum([title_ok, authors_ok, method_or_finding_ok, keywords_ok, summary_ok])
    return (
        title_ok
        and authors_ok
        and filled_key_fields >= config.extraction_gate_min_filled_key_fields
    )


def normalize_tokens_for_language(tokens: list[str], language: str) -> list[str]:
    """Normalize tokens in a language-aware but deterministic way."""
    normalized: list[str] = []
    for token in tokens:
        value = token.strip().lower()
        if language == "de":
            value = value.replace("ß", "ss")
        if language == "fr":
            value = "".join(
                c
                for c in unicodedata.normalize("NFKD", value)
                if not unicodedata.combining(c)
            )
        value = re.sub(r"[^\w\s]", " ", value)
        value = " ".join(value.split())
        if value:
            normalized.append(value)
    return normalized


def select_seeded_subset_indices(total_count: int, limit: int, seed: int) -> list[int]:
    """Return deterministic subset indices for reproducible experiments."""
    if total_count <= 0 or limit <= 0:
        return []
    if limit >= total_count:
        return list(range(total_count))
    rng = random.Random(seed)
    indices = list(range(total_count))
    rng.shuffle(indices)
    return sorted(indices[:limit])


def should_promote_from_subset(
    baseline_overall: float,
    candidate_overall: float,
    baseline_by_language: dict[str, float],
    candidate_by_language: dict[str, float],
) -> bool:
    """Promote only when overall improves and no language regresses."""
    if candidate_overall <= baseline_overall:
        return False
    for language, baseline_score in baseline_by_language.items():
        if candidate_by_language.get(language, -1.0) < baseline_score:
            return False
    return True
