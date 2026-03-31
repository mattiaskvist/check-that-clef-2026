from clef_retrieval.schemas import PaperEvidence, TweetEvidence


def test_tweet_evidence_has_query_text():
    evidence = TweetEvidence(query_text_for_embedding="abc")

    assert evidence.query_text_for_embedding == "abc"
    assert evidence.language == "unknown"
    assert evidence.keywords == []


def test_paper_evidence_has_required_fields():
    paper = PaperEvidence(
        pubkey="1",
        title="Title",
        authors="A",
        abstract="B",
        paper_text_for_embedding="T",
    )

    assert paper.pubkey == "1"
    assert paper.title == "Title"
    assert paper.paper_text_for_embedding == "T"
