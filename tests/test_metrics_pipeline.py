import unittest
from clef_pipeline.metrics import EvaluationMetrics


class MetricsPipelineTests(unittest.TestCase):
    def test_accumulates_stage_metrics_per_language_and_global(self):
        metrics = EvaluationMetrics(fusion_top_k=30)

        metrics.add_query(
            "en",
            true_pubkey="p1",
            stages={
                "dense": ["p1", "p2"],
                "sparse": ["p2", "p1"],
                "rrf": ["p1", "p2"],
                "final": ["p1", "p2"],
            },
        )
        metrics.add_query(
            "de",
            true_pubkey="px",
            stages={
                "dense": ["p2", "p3"],
                "sparse": ["px", "p2"],
                "rrf": ["p2", "px"],
                "final": ["px", "p2"],
            },
        )

        summary = metrics.summary()
        self.assertIn("en", summary["languages"])
        self.assertIn("de", summary["languages"])
        self.assertGreater(summary["global"]["final"]["mrr5"], 0.0)


if __name__ == "__main__":
    unittest.main()
