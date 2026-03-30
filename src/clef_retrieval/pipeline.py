"""Retrieval pipeline components."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from google.genai import errors
from pydantic import ValidationError

from .config import RetrievalConfig
from .gemini_client import GeminiService, build_embedding_input
from .query_policy import extraction_gate_passes, normalize_tokens_for_language
from .reranker import clip_top5
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
) -> list[str]:
    cfg = config or RetrievalConfig()
    pubkeys = [str(row.get("pubkey", "")) for row in metadata_rows]
    candidates = retrieve_top_pubkeys(
        query_embedding=query_embedding,
        paper_embeddings=paper_embeddings,
        pubkeys=pubkeys,
        k=top_k,
    )
    metadata_by_pubkey = {str(row.get("pubkey", "")): row for row in metadata_rows}
    rerank_impl = _lexical_rerank
    if evidence is not None:
        rerank_impl = lambda text, cands, row_map: weighted_signal_rerank(  # noqa: E731
            text,
            cands,
            row_map,
            evidence=evidence,
            config=cfg,
        )
    return stage2_rerank(
        tweet_text,
        candidates,
        rerank_fn=lambda text, cands: rerank_impl(text, cands, metadata_by_pubkey),
    )
