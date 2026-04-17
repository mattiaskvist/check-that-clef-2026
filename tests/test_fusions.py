"""Tests for the fusions module.

These tests run locally without GPU or Modal — all retrievers are mocked.
"""

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from clef_pipeline.fusions import (
    BaseFuser,
    FeatureGenerator,
    RandomForestFuser,
    RRFFuser,
)


# ---------------------------------------------------------------------------
# FeatureGenerator
# ---------------------------------------------------------------------------


class TestFeatureGenerator:
    def test_builds_correct_features_with_rrf(self):
        candidates = [0, 1, 2]
        dense_scores = [0.9, 0.5, 0.1]
        dense_ranked = [0, 1, 2]
        sparse_scores = [0.2, 0.8, 0.3]
        sparse_ranked = [1, 2, 0]
        rrf_scores = {0: 0.03, 1: 0.04, 2: 0.02}
        rrf_ranked = [1, 0, 2]

        features = FeatureGenerator.build_query_features(
            candidates,
            dense_scores,
            dense_ranked,
            sparse_scores,
            sparse_ranked,
            rrf_scores,
            rrf_ranked,
            include_rrf=True,
        )

        assert len(features) == 3
        assert len(features[0]) == 6  # 4 base + 2 rrf

        # Check doc 0 features
        assert features[0][0] == pytest.approx(0.9)  # dense_score
        assert features[0][1] == 0.0  # dense_rank (rank 0)
        assert features[0][2] == pytest.approx(0.2)  # sparse_score
        assert features[0][4] == pytest.approx(0.03)  # rrf_score

    def test_builds_correct_features_without_rrf(self):
        candidates = [0, 1]
        dense_scores = [0.9, 0.5]
        dense_ranked = [0, 1]
        sparse_scores = [0.2, 0.8]
        sparse_ranked = [1, 0]
        rrf_scores = {0: 0.03, 1: 0.04}
        rrf_ranked = [1, 0]

        features = FeatureGenerator.build_query_features(
            candidates,
            dense_scores,
            dense_ranked,
            sparse_scores,
            sparse_ranked,
            rrf_scores,
            rrf_ranked,
            include_rrf=False,
        )

        assert len(features) == 2
        assert len(features[0]) == 4  # no rrf features

    def test_empty_candidates(self):
        features = FeatureGenerator.build_query_features(
            [],
            [0.9],
            [0],
            [0.5],
            [0],
            {},
            [],
            include_rrf=True,
        )
        assert features == []

    def test_feature_names(self):
        names_with = FeatureGenerator.get_feature_names(include_rrf=True)
        names_without = FeatureGenerator.get_feature_names(include_rrf=False)
        assert len(names_with) == 6
        assert len(names_without) == 4
        assert "rrf_score" in names_with
        assert "rrf_score" not in names_without


# ---------------------------------------------------------------------------
# RRFFuser
# ---------------------------------------------------------------------------


class TestRRFFuser:
    def test_returns_correct_topk(self):
        fuser = RRFFuser()
        ranked_a = [0, 1, 2, 3, 4]
        ranked_b = [2, 3, 0, 1, 4]

        result = fuser.fuse([ranked_a, ranked_b], top_k=3)

        assert len(result) == 3
        assert all(isinstance(x, int) for x in result)

    def test_ignores_scores_and_lang(self):
        fuser = RRFFuser()
        ranked = [0, 1, 2]
        result_no_args = fuser.fuse([ranked, ranked], top_k=2)
        result_with_args = fuser.fuse(
            [ranked, ranked], scores_lists=[[0.5, 0.3, 0.1]], top_k=2, lang="en"
        )
        assert result_no_args == result_with_args

    def test_rrf_with_scores_returns_tuple(self):
        ranked_a = [0, 1, 2]
        ranked_b = [2, 1, 0]
        ids, scores = RRFFuser._rrf_with_scores([ranked_a, ranked_b], top_k=2)
        assert len(ids) == 2
        assert all(doc_id in scores for doc_id in ids)
        assert all(isinstance(v, float) for v in scores.values())

    def test_deterministic(self):
        fuser = RRFFuser()
        ranked = [5, 3, 1, 0, 2, 4]
        r1 = fuser.fuse([ranked, ranked], top_k=4)
        r2 = fuser.fuse([ranked, ranked], top_k=4)
        assert r1 == r2


# ---------------------------------------------------------------------------
# RandomForestFuser
# ---------------------------------------------------------------------------


def _make_mock_dense_retriever(n_docs: int = 20, n_queries: int = 10):
    """Create a mock dense retriever that returns deterministic results."""
    rng = np.random.RandomState(42)
    mock = MagicMock()
    mock.model_name = "mock/dense-v1"

    all_scores = {}
    all_ranks = {}

    for cache_name_suffix in ["en", "de", "fr"]:
        for prefix in ["queries_train_", "queries_"]:
            cache_name = f"{prefix}{cache_name_suffix}"
            scores_matrix = rng.rand(n_queries, n_docs).tolist()
            ranks_matrix = [list(np.argsort(s)[::-1]) for s in scores_matrix]
            all_scores[cache_name] = scores_matrix
            all_ranks[cache_name] = ranks_matrix

    def search_with_scores(query_idx, cache_name=None):
        return all_ranks[cache_name][query_idx], all_scores[cache_name][query_idx]

    def search(query_idx, cache_name=None):
        return all_ranks[cache_name][query_idx]

    mock.search_with_scores = search_with_scores
    mock.search = search
    mock.index_queries = MagicMock()
    return mock


def _make_mock_sparse_retriever(n_docs: int = 20, n_queries: int = 10):
    """Create a mock sparse retriever that returns deterministic results."""
    rng = np.random.RandomState(123)
    mock = MagicMock()
    mock.bm25_k1 = 2.5
    mock.bm25_b = 0.85

    all_scores = {}
    all_ranks = {}

    for cache_name_suffix in ["en", "de", "fr"]:
        for prefix in ["sparse_queries_train_", "sparse_queries_"]:
            cache_name = f"{prefix}{cache_name_suffix}"
            scores_matrix = rng.rand(n_queries, n_docs).tolist()
            ranks_matrix = [list(np.argsort(s)[::-1]) for s in scores_matrix]
            all_scores[cache_name] = scores_matrix
            all_ranks[cache_name] = ranks_matrix

    def search_with_scores(query_idx, cache_name=None):
        return all_ranks[cache_name][query_idx], all_scores[cache_name][query_idx]

    def search(query_idx, cache_name=None):
        return all_ranks[cache_name][query_idx]

    mock.search_with_scores = search_with_scores
    mock.search = search
    mock.index_queries = MagicMock()
    return mock


def _make_train_tweets(n_per_lang: int = 10, n_docs: int = 20):
    """Create synthetic train data."""
    rng = np.random.RandomState(7)
    pubkeys = [f"PK{i:04d}" for i in range(n_docs)]
    result = {}
    for lang in ["en", "de", "fr"]:
        tweets = []
        for i in range(n_per_lang):
            tweets.append(
                {
                    "text": f"tweet {lang} {i}",
                    "pubkey": pubkeys[rng.randint(0, n_docs)],
                }
            )
        result[lang] = tweets
    return result, pubkeys


class TestRandomForestFuser:
    def test_fuse_before_train_raises(self):
        fuser = RandomForestFuser()
        with pytest.raises(RuntimeError, match="not trained"):
            fuser.fuse(
                ranked_lists=[[0, 1], [1, 0]],
                scores_lists=[[0.9, 0.5], [0.3, 0.8]],
                top_k=2,
                lang="en",
            )

    def test_fuse_without_scores_raises(self):
        fuser = RandomForestFuser()
        fuser._trained = True
        fuser.models = {"en": MagicMock()}
        with pytest.raises(ValueError, match="requires scores_lists"):
            fuser.fuse(
                ranked_lists=[[0, 1], [1, 0]],
                scores_lists=None,
                top_k=2,
                lang="en",
            )

    def test_train_and_fuse_per_language(self):
        n_docs = 20
        dense = _make_mock_dense_retriever(n_docs=n_docs)
        sparse = _make_mock_sparse_retriever(n_docs=n_docs)
        tweets, pubkeys = _make_train_tweets(n_docs=n_docs)

        fuser = RandomForestFuser(global_model=False, candidate_top_k=10)
        fuser.train(
            dense_retriever=dense,
            sparse_retriever=sparse,
            train_tweets_by_lang=tweets,
            article_pubkeys=pubkeys,
            dense_model_name="mock/dense-v1",
            sparse_config={"k1": 2.5, "b": 0.85, "stemmer": "lancaster"},
        )

        assert fuser._trained
        assert "en" in fuser.models
        assert "de" in fuser.models
        assert "fr" in fuser.models

        # Fuse for each language
        for lang in ["en", "de", "fr"]:
            dense_ranks, dense_scores = dense.search_with_scores(
                0, cache_name=f"queries_{lang}"
            )
            sparse_ranks, sparse_scores = sparse.search_with_scores(
                0, cache_name=f"sparse_queries_{lang}"
            )

            result = fuser.fuse(
                ranked_lists=[dense_ranks, sparse_ranks],
                scores_lists=[dense_scores, sparse_scores],
                top_k=5,
                lang=lang,
            )
            assert len(result) == 5
            assert all(isinstance(x, int) for x in result)
            assert len(set(result)) == 5  # no duplicates

    def test_train_and_fuse_global(self):
        n_docs = 20
        dense = _make_mock_dense_retriever(n_docs=n_docs)
        sparse = _make_mock_sparse_retriever(n_docs=n_docs)
        tweets, pubkeys = _make_train_tweets(n_docs=n_docs)

        fuser = RandomForestFuser(global_model=True, candidate_top_k=10)
        fuser.train(
            dense_retriever=dense,
            sparse_retriever=sparse,
            train_tweets_by_lang=tweets,
            article_pubkeys=pubkeys,
            dense_model_name="mock/dense-v1",
            sparse_config={"k1": 2.5, "b": 0.85, "stemmer": "lancaster"},
        )

        assert fuser._trained
        assert "global" in fuser.models

        dense_ranks, dense_scores = dense.search_with_scores(0, cache_name="queries_en")
        sparse_ranks, sparse_scores = sparse.search_with_scores(
            0, cache_name="sparse_queries_en"
        )

        result = fuser.fuse(
            ranked_lists=[dense_ranks, sparse_ranks],
            scores_lists=[dense_scores, sparse_scores],
            top_k=5,
            lang="en",
        )
        assert len(result) == 5

    def test_save_and_load_roundtrip(self):
        n_docs = 20
        dense = _make_mock_dense_retriever(n_docs=n_docs)
        sparse = _make_mock_sparse_retriever(n_docs=n_docs)
        tweets, pubkeys = _make_train_tweets(n_docs=n_docs)

        fuser = RandomForestFuser(global_model=False, candidate_top_k=10)
        fuser.train(
            dense_retriever=dense,
            sparse_retriever=sparse,
            train_tweets_by_lang=tweets,
            article_pubkeys=pubkeys,
            dense_model_name="mock/dense-v1",
            sparse_config={"k1": 2.5, "b": 0.85, "stemmer": "lancaster"},
        )

        # Fuse before save
        dense_ranks, dense_scores = dense.search_with_scores(0, cache_name="queries_en")
        sparse_ranks, sparse_scores = sparse.search_with_scores(
            0, cache_name="sparse_queries_en"
        )
        result_before = fuser.fuse(
            ranked_lists=[dense_ranks, sparse_ranks],
            scores_lists=[dense_scores, sparse_scores],
            top_k=5,
            lang="en",
        )

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = fuser.save(tmpdir)
            assert os.path.exists(saved_path)

            fuser2 = RandomForestFuser(global_model=False, candidate_top_k=10)
            loaded = fuser2.load(
                cache_dir=tmpdir,
                dense_model_name="mock/dense-v1",
                sparse_config={"k1": 2.5, "b": 0.85, "stemmer": "lancaster"},
            )
            assert loaded is True
            assert fuser2._trained

            result_after = fuser2.fuse(
                ranked_lists=[dense_ranks, sparse_ranks],
                scores_lists=[dense_scores, sparse_scores],
                top_k=5,
                lang="en",
            )

        assert result_before == result_after

    def test_load_returns_false_when_no_cache(self):
        fuser = RandomForestFuser()
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = fuser.load(
                cache_dir=tmpdir,
                dense_model_name="nonexistent/model",
                sparse_config={"k1": 1.0, "b": 0.5, "stemmer": "lancaster"},
            )
        assert loaded is False

    def test_fingerprint_changes_with_config(self):
        fp1 = RandomForestFuser._compute_fingerprint(
            "model/a",
            {"k1": 2.5, "b": 0.85, "stemmer": "lancaster"},
            "train",
            {"n_estimators": 100},
            False,
            500,
        )
        fp2 = RandomForestFuser._compute_fingerprint(
            "model/b",
            {"k1": 2.5, "b": 0.85, "stemmer": "lancaster"},
            "train",
            {"n_estimators": 100},
            False,
            500,
        )
        fp3 = RandomForestFuser._compute_fingerprint(
            "model/a",
            {"k1": 1.5, "b": 0.75, "stemmer": "porter"},
            "train",
            {"n_estimators": 100},
            False,
            500,
        )
        fp4 = RandomForestFuser._compute_fingerprint(
            "model/a",
            {"k1": 2.5, "b": 0.85, "stemmer": "lancaster"},
            "train",
            {"n_estimators": 200},
            False,
            500,
        )

        assert fp1 != fp2  # different dense model
        assert fp1 != fp3  # different sparse config
        assert fp1 != fp4  # different RF params


# ---------------------------------------------------------------------------
# BaseFuser
# ---------------------------------------------------------------------------


class TestBaseFuser:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseFuser()
