from clef_retrieval.gemini_client import build_embedding_input
from clef_retrieval.schemas import TweetEvidence


def test_build_embedding_input_prefers_evidence_text():
    evidence = TweetEvidence(query_text_for_embedding="structured query")

    value = build_embedding_input(evidence, "raw tweet text")

    assert value == "structured query"


def test_build_embedding_input_falls_back_to_raw_tweet_when_empty():
    evidence = TweetEvidence(query_text_for_embedding="")

    value = build_embedding_input(evidence, "raw tweet text")

    assert value == "raw tweet text"
