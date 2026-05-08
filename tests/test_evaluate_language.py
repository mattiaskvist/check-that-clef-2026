import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clef_pipeline import evaluate_language


class EvaluateLanguageTests(unittest.TestCase):
    def test_split_indices_balances_workers(self):
        self.assertEqual(
            evaluate_language._split_indices(10, 4),
            [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]],
        )
        self.assertEqual(evaluate_language._split_indices(0, 4), [])

    def test_chunk_indices_uses_half_open_ranges(self):
        self.assertEqual(
            evaluate_language._chunk_indices(5, 2),
            [(0, 2), (2, 4), (4, 5)],
        )

    def test_base_payload_defaults_to_german_evaluation(self):
        args = evaluate_language.parse_args([])
        payload = evaluate_language._build_base_payload(args)

        self.assertEqual(payload["languages"], ["de"])
        self.assertEqual(payload["split"], "dev")
        self.assertEqual(payload["dense_model"], "harrier-27b")
        self.assertEqual(payload["fusion_method"], "rrf")

    def test_write_worker_payloads_splits_german_queries(self):
        fake_rows = [{"text": f"q{i}", "pubkey": f"p{i}"} for i in range(5)]
        args = evaluate_language.parse_args(["--num-workers", "2"])
        payload = evaluate_language._build_base_payload(args)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                evaluate_language, "load_query_split", return_value=fake_rows
            ):
                tasks = evaluate_language._write_worker_payloads(
                    base_payload=payload,
                    work_dir=tmpdir,
                    num_workers=2,
                    chunk_size=None,
                )

            self.assertEqual(len(tasks), 2)
            first_payload = json.loads(Path(tasks[0]["payload_path"]).read_text())
            second_payload = json.loads(Path(tasks[1]["payload_path"]).read_text())
            self.assertEqual(first_payload["query_indices"], [0, 1, 2])
            self.assertEqual(second_payload["query_indices"], [3, 4])

    def test_vertex_template_uses_requested_rf_qwen_8gpu_shape(self):
        repo_root = Path(__file__).resolve().parents[1]
        template = Path(repo_root, "infra/gcp/vertex-evaluation-de-8gpu.yaml")
        text = template.read_text()

        self.assertIn("machineType: a3-highgpu-8g", text)
        self.assertIn("acceleratorCount: 8", text)
        self.assertIn("--num-workers=8", text)
        self.assertIn("--gpu-ids=0,1,2,3,4,5,6,7", text)
        self.assertIn("--split=test", text)
        self.assertIn("--fusion-method=random_forest", text)
        self.assertIn("--fusion-top-k=200", text)
        self.assertIn("--reranker-model=qwen3-reranker-8b", text)
        self.assertIn("--export-submission-tsv", text)
        self.assertIn("--submission-output-dir=", text)
        self.assertNotIn("--metrics-output-file", text)


if __name__ == "__main__":
    unittest.main()
