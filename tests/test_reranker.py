from clef_retrieval.reranker import clip_top5


def test_clip_top5_returns_at_most_five_items():
    ranked = ["a", "b", "c", "d", "e", "f", "g"]

    assert clip_top5(ranked) == ["a", "b", "c", "d", "e"]


def test_clip_top5_keeps_shorter_input():
    ranked = ["a", "b"]

    assert clip_top5(ranked) == ["a", "b"]


def test_rerank_result_rationale_short_defaults_to_empty_list():
    from clef_retrieval.reranker import RerankResult

    result = RerankResult()

    assert result.rationale_short == []


def test_rerank_result_accepts_list_for_rationale_short():
    from clef_retrieval.reranker import RerankResult

    result = RerankResult(rationale_short=["keyword overlap", "title signal"])

    assert result.rationale_short == ["keyword overlap", "title signal"]
