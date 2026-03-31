from clef_retrieval.config import RetrievalConfig
from clef_retrieval.query_policy import (
    extraction_gate_passes,
    normalize_tokens_for_language,
    should_promote_from_subset,
)
from clef_retrieval.schemas import TweetEvidence


def test_extraction_gate_requires_minimum_key_strength():
    cfg = RetrievalConfig(
        extraction_gate_min_title_mentions=1,
        extraction_gate_min_authors=1,
        extraction_gate_min_filled_key_fields=3,
    )
    weak = TweetEvidence(
        query_text_for_embedding="q",
        candidate_title_mentions=["title only"],
        candidate_authors=[],
        keywords=[],
    )
    strong = TweetEvidence(
        query_text_for_embedding="q",
        claim_summary="summary",
        candidate_title_mentions=["title"],
        candidate_authors=["author"],
        method_terms=["method"],
    )

    assert extraction_gate_passes(weak, cfg) is False
    assert extraction_gate_passes(strong, cfg) is True


def test_language_normalization_for_en_de_fr():
    assert normalize_tokens_for_language([" The-Study! "], "en") == ["the study"]
    assert normalize_tokens_for_language(["Maßstab"], "de") == ["massstab"]
    assert normalize_tokens_for_language(["Étude clinique"], "fr") == ["etude clinique"]


def test_promotion_requires_overall_gain_and_no_language_regression():
    baseline_by_lang = {"en": 0.20, "de": 0.25, "fr": 0.30}
    improved_no_regression = {"en": 0.21, "de": 0.25, "fr": 0.31}
    regressed_de = {"en": 0.23, "de": 0.24, "fr": 0.33}

    assert (
        should_promote_from_subset(
            baseline_overall=0.25,
            candidate_overall=0.26,
            baseline_by_language=baseline_by_lang,
            candidate_by_language=improved_no_regression,
        )
        is True
    )
    assert (
        should_promote_from_subset(
            baseline_overall=0.25,
            candidate_overall=0.30,
            baseline_by_language=baseline_by_lang,
            candidate_by_language=regressed_de,
        )
        is False
    )
