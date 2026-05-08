import unittest
from clef_pipeline.main import _chunk_indices, _split_indices
from clef_pipeline.metrics import EvaluationMetrics


class MetricsPipelineTests(unittest.TestCase):
    def test_chunk_indices_returns_half_open_ranges(self):
        self.assertEqual(_chunk_indices(5, 2), [(0, 2), (2, 4), (4, 5)])
        self.assertEqual(_chunk_indices(0, 2), [])
        self.assertEqual(_chunk_indices(3, 0), [(0, 1), (1, 2), (2, 3)])

    def test_split_indices_balances_queries_across_workers(self):
        self.assertEqual(_split_indices(10, 3), [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(_split_indices(2, 4), [[0], [1]])
        self.assertEqual(_split_indices(0, 4), [])

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
        self.assertIn("r100", summary["languages"]["en"]["metrics"]["dense"])
        self.assertIn("r200", summary["languages"]["en"]["metrics"]["dense"])
        self.assertGreater(summary["global"]["final"]["mrr5"], 0.0)


if __name__ == "__main__":
    unittest.main()
