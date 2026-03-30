"""Retrieval pipeline components."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from google.genai import errors
from pydantic import ValidationError

from .gemini_client import GeminiService, build_embedding_input
from .reranker import clip_top5
from .retriever import retrieve_top_pubkeys
from .schemas import TweetEvidence


class QueryExtractionError(RuntimeError):
    """Raised when query evidence extraction fails unexpectedly."""


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
    try:
        evidence = service.extract_tweet_evidence(tweet_text)
        if not isinstance(evidence, TweetEvidence):
            raise QueryExtractionError("Gemini response did not parse as TweetEvidence")
        return build_embedding_input(evidence, tweet_text)
    except (errors.APIError, errors.ClientError, ValidationError, TypeError, ValueError, QueryExtractionError):
        return tweet_text


def predict_top5_with_embeddings(
    tweet_text: str,
    service: GeminiService,
    paper_embeddings,
    metadata_rows: list[dict],
    top_k: int,
) -> list[str]:
    query_text = build_query_embedding_text(tweet_text, service)
    query_embeddings = service.embed_texts([query_text])
    if not query_embeddings:
        return ensure_top5([])

    pubkeys = [str(row.get("pubkey", "")) for row in metadata_rows]
    candidates = retrieve_top_pubkeys(
        query_embedding=query_embeddings[0],
        paper_embeddings=paper_embeddings,
        pubkeys=pubkeys,
        k=top_k,
    )
    metadata_by_pubkey = {str(row.get("pubkey", "")): row for row in metadata_rows}
    return stage2_rerank(
        tweet_text,
        candidates,
        rerank_fn=lambda text, cands: _lexical_rerank(text, cands, metadata_by_pubkey),
    )
