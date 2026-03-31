from clef_retrieval.config import RetrievalConfig
from clef_retrieval.pipeline import weighted_signal_rerank
from clef_retrieval.schemas import TweetEvidence


def test_weighted_signal_ranking_prioritizes_title_author_over_lower_signals():
    config = RetrievalConfig()
    evidence = TweetEvidence(
        query_text_for_embedding="query",
        language="en",
        candidate_title_mentions=["climate model"],
        candidate_authors=["alice smith"],
        method_terms=["regression"],
        finding_terms=["warming trend"],
        keywords=["temperature"],
    )
    metadata_by_pubkey = {
        "title-author-match": {
            "pubkey": "title-author-match",
            "title": "Climate model advances",
            "authors": "Alice Smith",
            "abstract": "broad discussion",
            "keywords": [],
            "method_terms": [],
            "finding_terms": [],
        },
        "method-finding-match": {
            "pubkey": "method-finding-match",
            "title": "General weather paper",
            "authors": "Bob Jones",
            "abstract": "Regression reveals a warming trend in data.",
            "keywords": [],
            "method_terms": [],
            "finding_terms": [],
        },
        "keyword-only-match": {
            "pubkey": "keyword-only-match",
            "title": "Temperature statistics",
            "authors": "Carl Doe",
            "abstract": "temperature observations in urban areas",
            "keywords": [],
            "method_terms": [],
            "finding_terms": [],
        },
    }

    ranked = weighted_signal_rerank(
        tweet_text="irrelevant raw query",
        candidates=["keyword-only-match", "method-finding-match", "title-author-match"],
        metadata_by_pubkey=metadata_by_pubkey,
        evidence=evidence,
        config=config,
    )

    assert ranked[0] == "title-author-match"
    assert ranked.index("method-finding-match") < ranked.index("keyword-only-match")


def test_weighted_signal_ranking_applies_negative_constraints_penalty():
    config = RetrievalConfig()
    evidence = TweetEvidence(
        query_text_for_embedding="query",
        language="en",
        candidate_title_mentions=["climate model"],
        candidate_authors=["alice smith"],
        negative_constraints=["retracted"],
    )
    metadata_by_pubkey = {
        "penalized": {
            "pubkey": "penalized",
            "title": "Climate model",
            "authors": "Alice Smith",
            "abstract": "This study was retracted later.",
            "keywords": [],
            "method_terms": [],
            "finding_terms": [],
        },
        "clean": {
            "pubkey": "clean",
            "title": "Climate model",
            "authors": "Alice Smith",
            "abstract": "Stable reproducible study.",
            "keywords": [],
            "method_terms": [],
            "finding_terms": [],
        },
    }

    ranked = weighted_signal_rerank(
        tweet_text="query",
        candidates=["penalized", "clean"],
        metadata_by_pubkey=metadata_by_pubkey,
        evidence=evidence,
        config=config,
    )

    assert ranked[0] == "clean"
    assert ranked[1] == "penalized"
