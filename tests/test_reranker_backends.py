"""Tests for semantic reranker backend configuration and adapter contracts.

These tests lock decisions D-01, D-02, D-03 from 02-CONTEXT.md:
- D-01: Implement configurable support for both Jina Reranker v2 and BGE v2-m3
- D-02: Default reranker backend for Phase 2 experiments should be Jina Reranker v2
- D-03: Rerank the top 50 dense-retrieval candidates per query (rerank_top_k=50)
"""

import pytest

from clef_retrieval.config import RetrievalConfig


class TestRerankerBackendConfiguration:
    """Config supports jina_v2 and bge_v2_m3 backends with jina_v2 as default."""

    def test_config_defaults_to_jina_v2_backend(self):
        config = RetrievalConfig()

        assert config.reranker_backend == "jina_v2"

    def test_config_accepts_bge_v2_m3_backend(self):
        config = RetrievalConfig(reranker_backend="bge_v2_m3")

        assert config.reranker_backend == "bge_v2_m3"

    def test_config_rejects_unknown_backend(self):
        with pytest.raises(ValueError):
            RetrievalConfig(reranker_backend="invalid_backend")


class TestRerankDepthConfiguration:
    """Rerank depth is configurable with default 50 per D-03."""

    def test_config_defaults_rerank_top_k_to_50(self):
        config = RetrievalConfig()

        assert config.rerank_top_k == 50

    def test_config_accepts_custom_rerank_top_k(self):
        config = RetrievalConfig(rerank_top_k=100)

        assert config.rerank_top_k == 100

    def test_rerank_top_k_must_be_positive(self):
        with pytest.raises(ValueError):
            RetrievalConfig(rerank_top_k=0)


class TestTieBreakEpsilonConfiguration:
    """Tie-break epsilon threshold for semantic score equivalence."""

    def test_config_has_semantic_tie_epsilon_default(self):
        config = RetrievalConfig()

        assert hasattr(config, "semantic_tie_epsilon")
        assert config.semantic_tie_epsilon > 0

    def test_config_accepts_custom_tie_epsilon(self):
        config = RetrievalConfig(semantic_tie_epsilon=0.001)

        assert config.semantic_tie_epsilon == 0.001


class TestRerankerAdapterContract:
    """Backend adapter interface contract tests."""

    def test_semantic_reranker_protocol_exists(self):
        from clef_retrieval.reranker import SemanticReranker

        assert hasattr(SemanticReranker, "score_candidates")

    def test_get_reranker_returns_adapter_for_jina_backend(self):
        from clef_retrieval.reranker import get_reranker

        config = RetrievalConfig(reranker_backend="jina_v2")
        reranker = get_reranker(config)

        assert reranker is not None
        assert hasattr(reranker, "score_candidates")

    def test_get_reranker_returns_adapter_for_bge_backend(self):
        from clef_retrieval.reranker import get_reranker

        config = RetrievalConfig(reranker_backend="bge_v2_m3")
        reranker = get_reranker(config)

        assert reranker is not None
        assert hasattr(reranker, "score_candidates")

    def test_adapter_score_candidates_returns_pubkey_score_pairs(self):
        from clef_retrieval.reranker import get_reranker, MockReranker

        # Use mock adapter for unit test (no model loading)
        reranker = MockReranker()
        candidates = ["pub1", "pub2", "pub3"]
        metadata_by_pubkey = {
            "pub1": {"title": "Paper 1", "abstract": "Abstract 1"},
            "pub2": {"title": "Paper 2", "abstract": "Abstract 2"},
            "pub3": {"title": "Paper 3", "abstract": "Abstract 3"},
        }

        result = reranker.score_candidates(
            query="test query",
            candidates=candidates,
            metadata_by_pubkey=metadata_by_pubkey,
        )

        assert isinstance(result, list)
        assert len(result) == len(candidates)
        # Each result should be (pubkey, score) tuple
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)


class TestRerankerModelIds:
    """Backend-specific model identifiers are configured."""

    def test_jina_model_id_is_configured(self):
        config = RetrievalConfig()

        assert hasattr(config, "jina_reranker_model")
        assert "jina" in config.jina_reranker_model.lower()

    def test_bge_model_id_is_configured(self):
        config = RetrievalConfig()

        assert hasattr(config, "bge_reranker_model")
        assert "bge" in config.bge_reranker_model.lower()
