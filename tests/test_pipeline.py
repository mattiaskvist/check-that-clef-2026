from clef_retrieval.pipeline import (
    build_query_embedding_text,
    ensure_top5,
    predict_top5_with_embeddings,
    rank_from_query_embedding,
)
from clef_retrieval.schemas import TweetEvidence


class SuccessfulExtractor:
    def extract_tweet_evidence(self, tweet_text: str) -> TweetEvidence:
        return TweetEvidence(query_text_for_embedding=f"structured {tweet_text}")


class FailingExtractor:
    def extract_tweet_evidence(self, tweet_text: str):
        raise ValueError("parse failure")


class FakeService:
    def __init__(self):
        self.seen_queries: list[str] = []

    def extract_tweet_evidence(self, tweet_text: str) -> TweetEvidence:
        return TweetEvidence(query_text_for_embedding=tweet_text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.seen_queries.extend(texts)
        return [[1.0, 0.0] for _ in texts]


def test_ensure_top5_truncates_long_input():
    assert ensure_top5(["a", "b", "c", "d", "e", "f"]) == ["a", "b", "c", "d", "e"]


def test_ensure_top5_pads_short_input_with_last_seen_pubkey():
    assert ensure_top5(["1", "2"]) == ["1", "2", "2", "2", "2"]


def test_ensure_top5_returns_zeros_for_empty_input():
    assert ensure_top5([]) == ["0", "0", "0", "0", "0"]


def test_build_query_embedding_text_uses_structured_evidence():
    value = build_query_embedding_text("raw tweet", SuccessfulExtractor())

    assert value == "structured raw tweet"


def test_build_query_embedding_text_falls_back_to_raw_tweet_on_failure():
    value = build_query_embedding_text("raw tweet", FailingExtractor())

    assert value == "raw tweet"


def test_predict_top5_with_embeddings_returns_ranked_values_and_pads():
    service = FakeService()
    metadata_rows = [
        {"pubkey": "paper-a", "title": "Alpha", "abstract": "science claim"},
        {"pubkey": "paper-b", "title": "Beta", "abstract": "other details"},
    ]
    paper_embeddings = [[1.0, 0.0], [0.5, 0.5]]

    result = predict_top5_with_embeddings(
        tweet_text="science claim",
        service=service,
        paper_embeddings=paper_embeddings,
        metadata_rows=metadata_rows,
        top_k=2,
    )

    assert result == ["paper-a", "paper-b", "paper-b", "paper-b", "paper-b"]


def test_rank_from_query_embedding_returns_ranked_values_and_pads():
    metadata_rows = [
        {"pubkey": "paper-a", "title": "Alpha", "abstract": "science claim"},
        {"pubkey": "paper-b", "title": "Beta", "abstract": "other details"},
    ]
    paper_embeddings = [[1.0, 0.0], [0.5, 0.5]]

    result = rank_from_query_embedding(
        tweet_text="science claim",
        query_embedding=[1.0, 0.0],
        paper_embeddings=paper_embeddings,
        metadata_rows=metadata_rows,
        top_k=2,
    )

    assert result == ["paper-a", "paper-b", "paper-b", "paper-b", "paper-b"]
