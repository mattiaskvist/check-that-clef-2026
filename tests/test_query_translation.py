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
    def test_index_queries_translates_german_tweets_to_french(self):
        config = PipelineConfig(
            retrievers=[RetrieverConfig(name="dense")],
            reranker=RerankerConfig(name=None, enabled=False),
        )
        retriever = _FakeRetriever()
        pipeline = RetrievalPipeline(
            config=config,
            retrievers={"dense": retriever},
            reranker=None,
        )

        # Patch the module-level helper used by RetrievalPipeline.
        import clef_pipeline.pipeline as pipeline_module

        original = pipeline_module.translate_de_tweets_to_fr
        pipeline_module.translate_de_tweets_to_fr = lambda texts: [f"FR:{t}" for t in texts]
        try:
            pipeline.index_queries_for_language(lang="de", query_texts=["hallo welt"])
        finally:
            pipeline_module.translate_de_tweets_to_fr = original

        self.assertEqual(len(retriever.query_calls), 1)
        queries, _kwargs = retriever.query_calls[0]
        self.assertEqual(queries, ["FR:hallo welt"])

    def test_index_collection_translates_german_articles_to_french(self):
        config = PipelineConfig(
            retrievers=[RetrieverConfig(name="dense")],
            reranker=RerankerConfig(name=None, enabled=False),
        )
        retriever = _FakeRetriever()
        pipeline = RetrievalPipeline(
            config=config,
            retrievers={"dense": retriever},
            reranker=None,
        )

        import clef_pipeline.pipeline as pipeline_module

        original = pipeline_module.translate_de_articles_to_fr
        pipeline_module.translate_de_articles_to_fr = (
            lambda docs: [{**doc, "title": "FR_T", "abstract": "FR_A"} for doc in docs]
        )
        try:
            pipeline.index_collection(
                [
                    {
                        "pubkey": "p0",
                        "lang": "de",
                        "title": "DE_T",
                        "abstract": "DE_A",
                    }
                ]
            )
        finally:
            pipeline_module.translate_de_articles_to_fr = original

        self.assertEqual(pipeline.collection_documents[0]["title"], "FR_T")
        self.assertEqual(pipeline.collection_documents[0]["abstract"], "FR_A")
