import math
import os
import pickle
import tempfile
import unittest

import numpy as np

from full_pipeline.retrievers import HarrierRetriever, SparseRetriever


class _SortResult(list):
    def tolist(self):
        return list(self)


class _FakeTorch:
    @staticmethod
    def argsort(values, descending=False):
        indices = sorted(
            range(len(values)), key=lambda idx: values[idx], reverse=descending
        )
        return _SortResult(indices)

    @staticmethod
    def save(value, path):
        with open(path, "wb") as file:
            pickle.dump(value, file)

    @staticmethod
    def load(path, map_location=None, weights_only=False):
        with open(path, "rb") as file:
            return pickle.load(file)


class _FakeUtil:
    @staticmethod
    def cos_sim(query, embeddings):
        def _norm(vector):
            return math.sqrt(sum(v * v for v in vector))

        q_norm = _norm(query)
        scores = []
        for emb in embeddings:
            score = sum(q * d for q, d in zip(query, emb)) / (q_norm * _norm(emb))
            scores.append(score)
        return [scores]


class _CountingEncodeModel:
    def __init__(self):
        self.encode_calls = 0

    def encode(self, texts, **kwargs):
        self.encode_calls += 1
        return np.asarray(
            [[float(self.encode_calls), float(idx)] for idx in range(len(texts))],
            dtype=np.float32,
        )


class HarrierRetrieverRegressionTests(unittest.TestCase):
    def test_search_uses_named_query_cache(self):
        retriever = HarrierRetriever.__new__(HarrierRetriever)
        retriever.torch = _FakeTorch
        retriever.util = _FakeUtil
        retriever.embeddings = [[1.0, 0.0], [0.0, 1.0]]
        retriever.query_embeddings = [[1.0, 0.0]]
        retriever._query_embeddings_by_name = {
            "queries_de": [[1.0, 0.0]],
            "queries_en": [[0.0, 1.0]],
        }

        self.assertEqual(retriever.search(0, cache_name="queries_de"), [0, 1])
        self.assertEqual(retriever.search(0, cache_name="queries_en"), [1, 0])

    def test_cache_path_depends_on_text_fingerprint(self):
        retriever = HarrierRetriever.__new__(HarrierRetriever)
        retriever.model_name = "microsoft/harrier-oss-v1-27b"

        path_a = retriever._cache_path("/cache", "documents.pt", ["doc-a"])
        path_b = retriever._cache_path("/cache", "documents.pt", ["doc-b"])
        path_a_repeat = retriever._cache_path("/cache", "documents.pt", ["doc-a"])

        self.assertNotEqual(path_a, path_b)
        self.assertEqual(path_a, path_a_repeat)

    def test_force_recompute_rebuilds_document_cache(self):
        retriever = HarrierRetriever.__new__(HarrierRetriever)
        retriever.model_name = "microsoft/harrier-oss-v1-27b"
        retriever.batch_size = 2
        retriever.torch = _FakeTorch
        retriever.model = _CountingEncodeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = ["doc-a", "doc-b"]

            retriever.index(corpus, cache_dir=tmpdir, force_recompute=False)
            self.assertEqual(retriever.model.encode_calls, 1)
            self.assertEqual(retriever.embeddings[0][0], 1.0)

            retriever.index(corpus, cache_dir=tmpdir, force_recompute=False)
            self.assertEqual(retriever.model.encode_calls, 1)
            self.assertEqual(retriever.embeddings[0][0], 1.0)

            retriever.index(corpus, cache_dir=tmpdir, force_recompute=True)
            self.assertEqual(retriever.model.encode_calls, 2)
            self.assertEqual(retriever.embeddings[0][0], 2.0)

    def test_force_recompute_rebuilds_query_cache(self):
        retriever = HarrierRetriever.__new__(HarrierRetriever)
        retriever.model_name = "microsoft/harrier-oss-v1-27b"
        retriever.batch_size = 2
        retriever.torch = _FakeTorch
        retriever.model = _CountingEncodeModel()
        retriever._query_embeddings_by_name = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            queries = ["query-a", "query-b"]

            retriever.index_queries(
                queries,
                cache_dir=tmpdir,
                cache_name="queries_en",
                force_recompute=False,
            )
            self.assertEqual(retriever.model.encode_calls, 1)
            self.assertEqual(
                retriever._query_embeddings_by_name["queries_en"][0][0], 1.0
            )

            retriever.index_queries(
                queries,
                cache_dir=tmpdir,
                cache_name="queries_en",
                force_recompute=False,
            )
            self.assertEqual(retriever.model.encode_calls, 1)
            self.assertEqual(
                retriever._query_embeddings_by_name["queries_en"][0][0], 1.0
            )

            retriever.index_queries(
                queries,
                cache_dir=tmpdir,
                cache_name="queries_en",
                force_recompute=True,
            )
            self.assertEqual(retriever.model.encode_calls, 2)
            self.assertEqual(
                retriever._query_embeddings_by_name["queries_en"][0][0], 2.0
            )


class _FakeBM25:
    doc_len = [1, 1, 1]


class SparseRetrieverRegressionTests(unittest.TestCase):
    def test_search_uses_named_query_cache(self):
        retriever = SparseRetriever.__new__(SparseRetriever)
        retriever._query_rankings_by_name = {
            "sparse_de": np.asarray([[4, 1, 0], [2, 1, 0]], dtype=np.int32)
        }
        retriever._query_scores_by_name = {
            "sparse_de": np.asarray([[0.9, 0.5, 0.2], [1.2, 0.4, 0.1]], dtype=np.float32)
        }

        self.assertEqual(retriever.search(1, cache_name="sparse_de"), [2, 1, 0])
        ranks, scores = retriever.search_with_scores(0, cache_name="sparse_de")
        self.assertEqual(ranks, [4, 1, 0])
        for actual, expected in zip(scores, [0.9, 0.5, 0.2]):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_cache_path_depends_on_query_fingerprint_and_top_k(self):
        retriever = SparseRetriever.__new__(SparseRetriever)
        retriever.bm25_k1 = 2.5
        retriever.bm25_b = 0.85
        retriever._indexed_corpus_fingerprint = "deadbeefcafefeed"

        path_a = retriever._cache_path("/cache", "sparse_de", ["query a"], "de", 2000)
        path_b = retriever._cache_path("/cache", "sparse_de", ["query b"], "de", 2000)
        path_c = retriever._cache_path("/cache", "sparse_de", ["query a"], "de", 1000)
        path_a_repeat = retriever._cache_path(
            "/cache", "sparse_de", ["query a"], "de", 2000
        )

        self.assertNotEqual(path_a, path_b)
        self.assertNotEqual(path_a, path_c)
        self.assertEqual(path_a, path_a_repeat)

    def test_force_recompute_rebuilds_sparse_query_cache(self):
        retriever = SparseRetriever.__new__(SparseRetriever)
        retriever.bm25_k1 = 2.5
        retriever.bm25_b = 0.85
        retriever.bm25_model = _FakeBM25()
        retriever._indexed_corpus_fingerprint = "deadbeefcafefeed"
        retriever._query_rankings_by_name = {}
        retriever._query_scores_by_name = {}

        def _score_query(self, query, lang="auto", top_k=None):
            return (
                np.asarray([2, 1], dtype=np.int32),
                np.asarray([0.7, 0.3], dtype=np.float32),
            )

        retriever._score_query = _score_query.__get__(retriever, SparseRetriever)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = retriever._cache_path(
                tmpdir, "sparse_de", ["cached-query"], "de", 2
            )
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(
                cache_path,
                rankings=np.asarray([[0, 1]], dtype=np.int32),
                scores=np.asarray([[0.9, 0.8]], dtype=np.float32),
            )

            retriever.index_queries(
                ["cached-query"],
                lang="de",
                cache_dir=tmpdir,
                cache_name="sparse_de",
                top_k=2,
                force_recompute=False,
            )
            self.assertEqual(retriever.search(0, cache_name="sparse_de"), [0, 1])

            retriever.index_queries(
                ["cached-query"],
                lang="de",
                cache_dir=tmpdir,
                cache_name="sparse_de",
                top_k=2,
                force_recompute=True,
            )
            self.assertEqual(retriever.search(0, cache_name="sparse_de"), [2, 1])


if __name__ == "__main__":
    unittest.main()
