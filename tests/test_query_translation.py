import unittest


from clef_pipeline.pipeline import RetrievalPipeline
from clef_pipeline.pipeline_config import PipelineConfig, RetrieverConfig, RerankerConfig


class _FakeRetriever:
    def __init__(self):
        self.query_calls = []

    def index(self, corpus):
        pass

    def index_queries(self, queries, **kwargs):
        self.query_calls.append((list(queries), dict(kwargs)))

    def search(self, query_idx, cache_name=None):
        return []


class QueryTranslationTests(unittest.TestCase):
    def test_index_queries_does_not_translate_queries(self):
        config = PipelineConfig(
            retrievers=[RetrieverConfig(name="dense")],
            reranker=RerankerConfig(name=None, enabled=False),
            target_language="fr",
        )
        retriever = _FakeRetriever()
        pipeline = RetrievalPipeline(
            config=config,
            retrievers={"dense": retriever},
            reranker=None,
        )
        pipeline.index_queries_for_language(lang="de", query_texts=["hallo welt"])

        self.assertEqual(len(retriever.query_calls), 1)
        queries, _kwargs = retriever.query_calls[0]
        self.assertEqual(queries, ["hallo welt"])

    def test_reranker_translates_candidate_articles_only(self):
        config = PipelineConfig(
            retrievers=[RetrieverConfig(name="dense")],
            reranker=RerankerConfig(name="nemotron", enabled=True),
            use_fusion=False,
            fusion_top_k=3,
            final_top_k=3,
            target_language="fr",
        )

        class _SpyReranker:
            def __init__(self):
                self.last_corpus = None

            def preprocess_corpus(self, corpus):
                return [f"{doc['title']} {doc['abstract']}" for doc in corpus]

            def rerank(self, query, doc_indices, corpus):
                self.last_corpus = list(corpus)
                return [(idx, 1.0) for idx in doc_indices]

        spy = _SpyReranker()
        retriever = _FakeRetriever()
        pipeline = RetrievalPipeline(
            config=config,
            retrievers={"dense": retriever},
            reranker=spy,
        )
        pipeline.index_collection(
            [
                {"pubkey": "p0", "lang": "en", "title": "T0", "abstract": "A0"},
                {"pubkey": "p1", "lang": "de", "title": "T1", "abstract": "A1"},
                {"pubkey": "p2", "lang": "fr", "title": "T2", "abstract": "A2"},
            ]
        )

        import clef_pipeline.pipeline as pipeline_module

        original_translate = pipeline_module.translate_texts
        pipeline_module.translate_texts = (
            lambda texts, source_language, target_language: [
                f"{target_language}:{source_language}:{t}" for t in texts
            ]
        )
        try:
            pipeline._apply_reranker("q", [1])
        finally:
            pipeline_module.translate_texts = original_translate

        self.assertIsNotNone(spy.last_corpus)
        self.assertEqual(spy.last_corpus[1], "fr:de:T1 A1")
        self.assertEqual(spy.last_corpus[0], "T0 A0")
        self.assertEqual(spy.last_corpus[2], "T2 A2")
