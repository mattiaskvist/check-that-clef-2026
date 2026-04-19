import csv
import os
import tempfile
import unittest

from clef_pipeline.submission import (
    normalize_split,
    modal_volume_download_command,
    submission_volume_remote_dir,
    write_submission_tsv_files,
)


class SubmissionExportTests(unittest.TestCase):
    def test_write_submission_tsv_files_writes_expected_schema(self):
        payload = {
            "en": [{"index": 42, "preds": ["p1", "p2", "p3", "p4", "p5"]}],
            "de": [{"index": 7, "preds": ["d1", "d2", "d3", "d4", "d5"]}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_submission_tsv_files(payload, tmpdir)

            self.assertEqual(
                paths,
                [
                    os.path.join(tmpdir, "predictions_de.tsv"),
                    os.path.join(tmpdir, "predictions_en.tsv"),
                ],
            )

            with open(paths[1], newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file, delimiter="\t"))

            self.assertEqual(rows[0], ["index", "preds"])
            self.assertEqual(rows[1], ["42", "['p1', 'p2', 'p3', 'p4', 'p5']"])

    def test_write_submission_tsv_files_requires_pred_list(self):
        payload = {"en": [{"index": 1, "preds": "not-a-list"}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(TypeError):
                write_submission_tsv_files(payload, tmpdir)

    def test_write_submission_tsv_files_requires_exactly_5_predictions(self):
        payload = {"en": [{"index": 1, "preds": ["p1", "p2", "p3", "p4"]}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_submission_tsv_files(payload, tmpdir)

    def test_submission_volume_remote_dir_maps_mount_path(self):
        remote_dir = submission_volume_remote_dir(
            "/cache/embeddings/submissions/dev-20260416-191700",
            "/cache/embeddings",
        )
        self.assertEqual(remote_dir, "/submissions/dev-20260416-191700")

    def test_submission_volume_remote_dir_rejects_path_outside_mount(self):
        with self.assertRaises(ValueError):
            submission_volume_remote_dir(
                "/tmp/submissions/dev-20260416-191700", "/cache/embeddings"
            )

    def test_modal_volume_download_command(self):
        command = modal_volume_download_command(
            volume_name="checkthat-embedding-cache",
            remote_dir="/submissions/dev-20260416-191700",
            local_destination="submissions",
        )
        self.assertEqual(
            command,
            "uv run modal volume get checkthat-embedding-cache /submissions/dev-20260416-191700 submissions",
        )

    def test_normalize_split_accepts_train_dev_test(self):
        self.assertEqual(normalize_split("train"), "train")
        self.assertEqual(normalize_split("dev"), "dev")
        self.assertEqual(normalize_split("test"), "test")

    def test_normalize_split_rejects_unknown(self):
        with self.assertRaises(ValueError):
            normalize_split("validation")


if __name__ == "__main__":
    unittest.main()
