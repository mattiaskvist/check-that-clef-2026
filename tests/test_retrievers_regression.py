import math
import unittest

from full_pipeline.retrievers import HarrierRetriever


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


if __name__ == "__main__":
    unittest.main()
