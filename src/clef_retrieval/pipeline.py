"""Retrieval pipeline components."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from google.genai import errors
from pydantic import ValidationError

from .config import RetrievalConfig
from .disambiguation import disambiguate_title_collisions
from .gemini_client import GeminiService, build_embedding_input
from .query_policy import extraction_gate_passes, normalize_tokens_for_language
from .reranker import clip_top5, semantic_primary_sort
from .retriever import retrieve_top_pubkeys
from .schemas import TweetEvidence


class QueryExtractionError(RuntimeError):
    """Raised when query evidence extraction fails unexpectedly."""


ExtractionOutcome = Literal["parsed+accepted", "parsed+rejected", "error->fallback"]


def ensure_top5(pubkeys: Sequence[str]) -> list[str]:
    values = list(pubkeys[:5])
    if not values:
        return ["0"] * 5
    while len(values) < 5:
        values.append(values[-1])
    return values


def stage1_retrieve(
    tweet_text: str,
    retrieve_fn: Callable[[str], Sequence[str]],
) -> list[str]:
    return list(retrieve_fn(tweet_text))


def _lexical_rerank(tweet_text: str, candidates: Sequence[str], metadata_by_pubkey: dict[str, dict]) -> list[str]:
    tweet_tokens = {token for token in tweet_text.lower().split() if len(token) > 2}

    def score(pubkey: str) -> tuple[int, str]:
        row = metadata_by_pubkey.get(pubkey, {})
        paper_text = f"{row.get('title', '')} {row.get('abstract', '')}".lower()
        paper_tokens = set(paper_text.split())
        return len(tweet_tokens & paper_tokens), pubkey

    return [pubkey for pubkey, _ in sorted(((p, score(p)) for p in candidates), key=lambda item: item[1], reverse=True)]


def _as_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item).strip()]
    return []


def _normalize_evidence_terms(evidence: TweetEvidence, config: RetrievalConfig) -> tuple[list[str], list[str], list[str], list[str], str]:
    language = evidence.language if evidence.language in {"en", "de", "fr"} else "en"
    if config.language_normalization_mode != "strict":
        return [], [], [], [], language

    title_author_terms = normalize_tokens_for_language(
        [*evidence.candidate_title_mentions, *evidence.candidate_authors],
        language,
    )
    method_finding_terms = normalize_tokens_for_language(
        [*evidence.method_terms, *evidence.finding_terms],
        language,
    )
    keyword_terms = normalize_tokens_for_language(evidence.keywords, language)
    negative_terms = normalize_tokens_for_language(evidence.negative_constraints, language)
    return title_author_terms, method_finding_terms, keyword_terms, negative_terms, language


def weighted_signal_rerank(
    tweet_text: str,
    candidates: Sequence[str],
    metadata_by_pubkey: dict[str, dict],
    evidence: TweetEvidence,
    config: RetrievalConfig,
) -> list[str]:
    title_author_terms, method_finding_terms, keyword_terms, negative_terms, language = _normalize_evidence_terms(evidence, config)
    candidate_positions = {pubkey: idx for idx, pubkey in enumerate(candidates)}

    def score(pubkey: str) -> tuple[float, int]:
        row = metadata_by_pubkey.get(pubkey, {})
        doc_tokens = normalize_tokens_for_language(
            [
                str(row.get("title", "")),
                str(row.get("authors", "")),
                str(row.get("abstract", "")),
                *_as_string_list(row.get("keywords", [])),
                *_as_string_list(row.get("method_terms", [])),
                *_as_string_list(row.get("finding_terms", [])),
            ],
            language,
        )
        doc_text = " ".join(doc_tokens)
        title_author_matches = sum(1 for term in title_author_terms if term and term in doc_text)
        method_finding_matches = sum(1 for term in method_finding_terms if term and term in doc_text)
        keyword_matches = sum(1 for term in keyword_terms if term and term in doc_text)
        negative_matches = sum(1 for term in negative_terms if term and term in doc_text)

        weighted_score = (
            config.weight_title_author * title_author_matches
            + config.weight_method_finding * method_finding_matches
            + config.weight_keywords * keyword_matches
            - ((config.weight_title_author + config.weight_method_finding) * negative_matches)
        )
        return weighted_score, -candidate_positions.get(pubkey, 0)

    return [pubkey for pubkey, _ in sorted(((p, score(p)) for p in candidates), key=lambda item: item[1], reverse=True)]


def _compute_weighted_scores(
    candidates: Sequence[str],
    metadata_by_pubkey: dict[str, dict],
    evidence: TweetEvidence | None,
    config: RetrievalConfig,
) -> dict[str, float]:
    """Compute Phase 1 weighted signal scores for all candidates.

    Used as tie-break signal when semantic scores are equal.

    Returns:
        Dict mapping pubkey to weighted score
    """
    if evidence is None:
        # No evidence = lexical fallback (simple overlap score)
        return {pubkey: 0.0 for pubkey in candidates}

    title_author_terms, method_finding_terms, keyword_terms, negative_terms, language = _normalize_evidence_terms(evidence, config)

    scores = {}
    for pubkey in candidates:
        row = metadata_by_pubkey.get(pubkey, {})
        doc_tokens = normalize_tokens_for_language(
            [
                str(row.get("title", "")),
                str(row.get("authors", "")),
                str(row.get("abstract", "")),
                *_as_string_list(row.get("keywords", [])),
                *_as_string_list(row.get("method_terms", [])),
                *_as_string_list(row.get("finding_terms", [])),
            ],
            language,
        )
        doc_text = " ".join(doc_tokens)
        title_author_matches = sum(1 for term in title_author_terms if term and term in doc_text)
        method_finding_matches = sum(1 for term in method_finding_terms if term and term in doc_text)
        keyword_matches = sum(1 for term in keyword_terms if term and term in doc_text)
        negative_matches = sum(1 for term in negative_terms if term and term in doc_text)

        weighted_score = (
            config.weight_title_author * title_author_matches
            + config.weight_method_finding * method_finding_matches
            + config.weight_keywords * keyword_matches
            - ((config.weight_title_author + config.weight_method_finding) * negative_matches)
        )
        scores[pubkey] = weighted_score

    return scores


def stage2_rerank(
    tweet_text: str,
    candidates: Sequence[str],
    rerank_fn: Callable[[str, Sequence[str]], Sequence[str]] | None = None,
) -> list[str]:
    if rerank_fn is None:
        return ensure_top5(clip_top5(list(candidates)))
    reranked = list(rerank_fn(tweet_text, candidates))
    return ensure_top5(clip_top5(reranked))


def predict_top5(
    tweet_text: str,
    retrieve_fn: Callable[[str], Sequence[str]],
    rerank_fn: Callable[[str, Sequence[str]], Sequence[str]] | None = None,
) -> list[str]:
    candidates = stage1_retrieve(tweet_text, retrieve_fn)
    return stage2_rerank(tweet_text, candidates, rerank_fn=rerank_fn)


def build_query_embedding_text(tweet_text: str, service: GeminiService) -> str:
    query_text, _, _ = build_query_embedding_text_with_metadata(tweet_text, service)
    return query_text


def build_query_embedding_text_with_metadata(
    tweet_text: str,
    service: GeminiService,
    config: RetrievalConfig | None = None,
) -> tuple[str, ExtractionOutcome, TweetEvidence | None]:
    cfg = config or RetrievalConfig()
    try:
        evidence = service.extract_tweet_evidence(tweet_text)
        if not isinstance(evidence, TweetEvidence):
            raise QueryExtractionError("Gemini response did not parse as TweetEvidence")
        if extraction_gate_passes(evidence, cfg):
            return build_embedding_input(evidence, tweet_text), "parsed+accepted", evidence
        return tweet_text, "parsed+rejected", evidence
    except (errors.APIError, errors.ClientError, ValidationError, TypeError, ValueError, QueryExtractionError):
        return tweet_text, "error->fallback", None


def predict_top5_with_embeddings(
    tweet_text: str,
    service: GeminiService,
    paper_embeddings,
    metadata_rows: list[dict],
    top_k: int,
) -> list[str]:
    cfg = getattr(service, "config", RetrievalConfig())
    query_text, _, evidence = build_query_embedding_text_with_metadata(tweet_text, service, cfg)
    query_embeddings = service.embed_texts([query_text])
    if not query_embeddings:
        return ensure_top5([])

    return rank_from_query_embedding(
        tweet_text=tweet_text,
        query_embedding=query_embeddings[0],
        paper_embeddings=paper_embeddings,
        metadata_rows=metadata_rows,
        top_k=top_k,
        evidence=evidence,
        config=cfg,
    )


def rank_from_query_embedding(
    tweet_text: str,
    query_embedding,
    paper_embeddings,
    metadata_rows: list[dict],
    top_k: int,
    evidence: TweetEvidence | None = None,
    config: RetrievalConfig | None = None,
    semantic_reranker=None,
) -> list[str]:
    """Rank papers for a query embedding using dense retrieval + semantic reranking.

    Phase 2 behavior (D-05, D-06):
    1. Retrieve top_k candidates from dense similarity
    2. Apply semantic reranking to top rerank_top_k candidates
    3. Sort by semantic score (primary), weighted score (tie-break), original index (fallback)
    4. Return top 5 results

    Args:
        tweet_text: Original tweet text for reranking
        query_embedding: Query embedding vector
        paper_embeddings: Paper embedding matrix
        metadata_rows: List of paper metadata dicts
        top_k: Number of dense retrieval candidates
        evidence: Extracted tweet evidence (optional, for weighted scoring)
        config: Retrieval configuration
        semantic_reranker: Optional SemanticReranker instance (uses MockReranker if None)

    Returns:
        List of 5 pubkeys in ranked order
    """
    cfg = config or RetrievalConfig()
    pubkeys = [str(row.get("pubkey", "")) for row in metadata_rows]

    # Stage 1: Dense retrieval
    candidates = retrieve_top_pubkeys(
        query_embedding=query_embedding,
        paper_embeddings=paper_embeddings,
        pubkeys=pubkeys,
        k=top_k,
    )

    if not candidates:
        return ensure_top5([])

    metadata_by_pubkey = {str(row.get("pubkey", "")): row for row in metadata_rows}

    # Stage 2: Semantic reranking on top rerank_top_k candidates
    rerank_candidates = candidates[: cfg.rerank_top_k]
    remaining_candidates = candidates[cfg.rerank_top_k :]

    # Get semantic scores
    if semantic_reranker is not None:
        semantic_scores = dict(
            semantic_reranker.score_candidates(
                query=tweet_text,
                candidates=rerank_candidates,
                metadata_by_pubkey=metadata_by_pubkey,
            )
        )
    else:
        # Fallback: use mock scores based on dense retrieval order
        # This preserves Phase 1 behavior when no semantic reranker is available
        from .reranker import MockReranker

        mock = MockReranker()
        semantic_scores = dict(
            mock.score_candidates(
                query=tweet_text,
                candidates=rerank_candidates,
                metadata_by_pubkey=metadata_by_pubkey,
            )
        )

    # Get weighted scores for tie-breaking
    weighted_scores = _compute_weighted_scores(
        candidates=rerank_candidates,
        metadata_by_pubkey=metadata_by_pubkey,
        evidence=evidence,
        config=cfg,
    )

    # Build candidates with all scores: (pubkey, semantic, weighted, original_index)
    candidates_with_scores = [
        (
            pubkey,
            semantic_scores.get(pubkey, 0.0),
            weighted_scores.get(pubkey, 0.0),
            idx,
        )
        for idx, pubkey in enumerate(rerank_candidates)
    ]

    # Sort with semantic-primary, weighted tie-break, original index fallback
    sorted_candidates = semantic_primary_sort(
        candidates_with_scores, cfg.semantic_tie_epsilon
    )

    # Extract pubkeys in sorted order
    reranked = [item[0] for item in sorted_candidates]

    # Phase 3: Disambiguation for duplicate titles (D-01 through D-06)
    if evidence is not None:
        reranked = disambiguate_title_collisions(
            candidates=reranked,
            evidence=evidence,
            metadata_by_pubkey=metadata_by_pubkey,
            config=cfg,
        )

    # Append remaining candidates (not reranked) in original dense order
    final_order = reranked + remaining_candidates

    return ensure_top5(clip_top5(final_order))
