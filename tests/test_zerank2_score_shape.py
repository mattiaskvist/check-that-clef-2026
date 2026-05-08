"""Regression tests for zerank-2 predict output normalization."""

import unittest

import numpy as np

from clef_pipeline.rerankers import Zerank2Reranker


class Zerank2ScoreShapeTests(unittest.TestCase):
    def test_list_float_passthrough(self):
        raw = [0.1, 0.9, 0.5]
        self.assertEqual(Zerank2Reranker._scores_from_predict(raw, 3), [0.1, 0.9, 0.5])

    def test_1d_numpy(self):
        raw = np.array([1.0, 2.0, 3.0])
        self.assertEqual(Zerank2Reranker._scores_from_predict(raw, 3), [1.0, 2.0, 3.0])

    def test_2d_logits_uses_softmax_last_class(self):
        # Simulates (n, 2) logits: row i prefers class 1 when logit1 > logit0
        raw = np.array([[-1.0, 1.0], [2.0, -1.0], [0.0, 0.0]])
        scores = Zerank2Reranker._scores_from_predict(raw, 3)
        self.assertEqual(len(scores), 3)
        self.assertGreater(scores[0], scores[1])  # high P(class1) vs low

    def test_2d_single_column(self):
        raw = np.array([[0.3], [0.7]])
        self.assertEqual(Zerank2Reranker._scores_from_predict(raw, 2), [0.3, 0.7])


if __name__ == "__main__":
    unittest.main()
