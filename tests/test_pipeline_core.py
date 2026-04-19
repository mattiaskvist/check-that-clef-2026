import unittest
from clef_pipeline.pipeline import RetrievalPipeline
from clef_pipeline.pipeline_config import (
    PipelineConfig,
    RerankerConfig,
    RetrieverConfig,
)


class _FakeRetriever:
    def __init__(self, ranked_indices, scores=None):
        self._ranked_indices = ranked_indices
        self._scores = (
            list(scores)
            if scores is not None
            else [float(len(ranked_indices) - i) for i in range(len(ranked_indices))]
        )
        self.indexed = False
        self.query_calls = []

    def index(self, corpus):
        self.indexed = True

    def index_queries(self, queries, **kwargs):
        self.query_calls.append((tuple(queries), kwargs))

    def search(self, query_idx, cache_name=None):
        return list(self._ranked_indices)

    def search_with_scores(self, query_idx, cache_name=None):
        return list(self._ranked_indices), list(self._scores)


class _FakeReranker:
    def preprocess_corpus(self, corpus):
        return [f"{doc['title']} {doc['abstract']}" for doc in corpus]

    def rerank(self, query, doc_indices, corpus):
        # Deterministic rerank order for tests.
        return [
            (idx, float(100 - pos)) for pos, idx in enumerate(reversed(doc_indices))
        ]


class _FakeFusionModel:
    def __init__(self):
        self.calls = []

    def fuse(self, ranked_lists, scores_lists=None, top_k=10, lang=None):
        self.calls.append(
            {
                "ranked_lists": ranked_lists,
                "scores_lists": scores_lists,
                "top_k": top_k,
                "lang": lang,
            }
        )
        return list(ranked_lists[1])[:top_k]


class PipelineCoreTests(unittest.TestCase):
    def test_search_text_returns_top5_and_stage_outputs(self):
        corpus = [
            {"pubkey": "p0", "title": "t0", "abstract": "a0"},
            {"pubkey": "p1", "title": "t1", "abstract": "a1"},
            {"pubkey": "p2", "title": "t2", "abstract": "a2"},
            {"pubkey": "p3", "title": "t3", "abstract": "a3"},
            {"pubkey": "p4", "title": "t4", "abstract": "a4"},
            {"pubkey": "p5", "title": "t5", "abstract": "a5"},
        ]

        config = PipelineConfig(
            retrievers=[RetrieverConfig(name="dense"), RetrieverConfig(name="sparse")],
            reranker=RerankerConfig(name="nemotron", enabled=True),
            use_fusion=True,
            fusion_top_k=6,
            final_top_k=5,
        )
        pipeline = RetrievalPipeline(
            config=config,
            retrievers={
                "dense": _FakeRetriever([0, 1, 2, 3, 4, 5]),
                "sparse": _FakeRetriever([5, 4, 3, 2, 1, 0]),
            },
            reranker=_FakeReranker(),
        )

        pipeline.index_collection(corpus)
        result = pipeline.search_text("hello world", lang="en")

        self.assertEqual(len(result["preds"]), 5)
        self.assertIn("dense", result["stages"])
        self.assertIn("sparse", result["stages"])
        self.assertIn("rrf", result["stages"])
        self.assertIn("final", result["stages"])

    def test_search_text_without_reranker_uses_fusion_output(self):
        corpus = [
            {"pubkey": "p0", "title": "t0", "abstract": "a0"},
            {"pubkey": "p1", "title": "t1", "abstract": "a1"},
            {"pubkey": "p2", "title": "t2", "abstract": "a2"},
            {"pubkey": "p3", "title": "t3", "abstract": "a3"},
            {"pubkey": "p4", "title": "t4", "abstract": "a4"},
        ]

        config = PipelineConfig(
            retrievers=[RetrieverConfig(name="dense"), RetrieverConfig(name="sparse")],
            reranker=RerankerConfig(name=None, enabled=False),
            use_fusion=True,
            fusion_top_k=5,
            final_top_k=5,
        )
        pipeline = RetrievalPipeline(
            config=config,
            retrievers={
                "dense": _FakeRetriever([0, 1, 2, 3, 4]),
                "sparse": _FakeRetriever([4, 3, 2, 1, 0]),
            },
            reranker=None,
        )

        pipeline.index_collection(corpus)
        result = pipeline.search_text("hello world", lang="en")

        self.assertEqual(result["stages"]["final"], result["stages"]["rrf"])

    def test_search_text_keeps_base_language_for_sparse_translation(self):
        corpus = [
            {"pubkey": "p0", "title": "t0", "abstract": "a0"},
            {"pubkey": "p1", "title": "t1", "abstract": "a1"},
        ]
        dense = _FakeRetriever([0, 1])
        sparse = _FakeRetriever([1, 0])

        config = PipelineConfig(
            retrievers=[RetrieverConfig(name="dense"), RetrieverConfig(name="sparse")],
            reranker=RerankerConfig(name=None, enabled=False),
            use_fusion=True,
            fusion_top_k=2,
            final_top_k=2,
        )
        pipeline = RetrievalPipeline(
            config=config,
            retrievers={"dense": dense, "sparse": sparse},
            reranker=None,
        )

        pipeline.index_collection(corpus)
        pipeline.search_text("bonjour", lang="fr")
        pipeline.search_text("hello", lang="auto")

        self.assertEqual(sparse.query_calls[0][1].get("lang"), "fr")
        self.assertEqual(sparse.query_calls[1][1].get("lang"), "auto")

    def test_random_forest_fusion_passes_scores_and_lang(self):
        corpus = [
            {"pubkey": "p0", "title": "t0", "abstract": "a0"},
            {"pubkey": "p1", "title": "t1", "abstract": "a1"},
            {"pubkey": "p2", "title": "t2", "abstract": "a2"},
        ]
        dense = _FakeRetriever([0, 1, 2], scores=[0.9, 0.6, 0.2])
        sparse = _FakeRetriever([2, 1, 0], scores=[0.1, 0.7, 0.8])

        config = PipelineConfig(
            retrievers=[RetrieverConfig(name="dense"), RetrieverConfig(name="sparse")],
            reranker=RerankerConfig(name=None, enabled=False),
            use_fusion=True,
            fusion_method="random_forest",
            fusion_top_k=3,
            final_top_k=3,
        )
        pipeline = RetrievalPipeline(
            config=config,
            retrievers={"dense": dense, "sparse": sparse},
            reranker=None,
        )
        fake_fuser = _FakeFusionModel()
        pipeline.fuser = fake_fuser

        pipeline.index_collection(corpus)
        pipeline.index_queries_for_language(lang="de", query_texts=["bonjour"])
        result = pipeline.search_cached_query(
            query_idx=0, query_text="bonjour", lang="de"
        )

        self.assertEqual(len(fake_fuser.calls), 1)
        call = fake_fuser.calls[0]
        self.assertEqual(call["top_k"], 3)
        self.assertEqual(call["lang"], "de")
        self.assertIsNotNone(call["scores_lists"])
        self.assertEqual(call["scores_lists"][0], [0.9, 0.6, 0.2])
        self.assertEqual(call["scores_lists"][1], [0.1, 0.7, 0.8])
        self.assertEqual(result["stages"]["rrf"], ["p2", "p1", "p0"])


if __name__ == "__main__":
    unittest.main()
