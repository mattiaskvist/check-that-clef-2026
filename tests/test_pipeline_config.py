import unittest
from clef_pipeline.pipeline_config import build_pipeline_config


class PipelineConfigTests(unittest.TestCase):
    def test_demo_profile_uses_harrier_270m_with_sparse_and_reranker(self):
        config = build_pipeline_config("demo")

        self.assertEqual([r.name for r in config.retrievers], ["harrier-270m", "sparse"])
        self.assertTrue(config.use_fusion)
        self.assertEqual(config.reranker.name, "nemotron")
        self.assertTrue(config.reranker.enabled)
        self.assertEqual(config.final_top_k, 5)

    def test_profile_without_reranker_is_supported(self):
        config = build_pipeline_config("retrieval-only")

        self.assertFalse(config.reranker.enabled)
        self.assertIsNone(config.reranker.name)


if __name__ == "__main__":
    unittest.main()
