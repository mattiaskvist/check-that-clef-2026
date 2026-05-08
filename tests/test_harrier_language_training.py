import tempfile
import unittest
from pathlib import Path

from clef_training.train_harrier_language import (
    build_ir_evaluator_payload,
    build_training_kwargs,
    default_language_paths,
    language_name,
    make_triplet_record,
    normalize_language,
    query_prompt_for_language,
    resolve_resume_checkpoint,
)


class HarrierLanguageTrainingTests(unittest.TestCase):
    def test_language_aliases_normalize_to_dataset_configs(self):
        self.assertEqual(normalize_language("German"), "de")
        self.assertEqual(normalize_language("english"), "en")
        self.assertEqual(language_name("fr"), "French")

    def test_german_prompt_mentions_german_claims(self):
        prompt = query_prompt_for_language("de")

        self.assertIn("German claim", prompt)
        self.assertTrue(prompt.endswith("\nQuery: "))

    def test_default_paths_are_language_scoped(self):
        paths = default_language_paths("/mnt/data", "de")

        self.assertEqual(
            paths["triplets_path"], "/mnt/data/harrier_de_hard_negatives.jsonl"
        )
        self.assertEqual(
            paths["checkpoint_dir"], "/mnt/data/harrier-27b-de-checkpoints"
        )
        self.assertEqual(paths["output_dir"], "/mnt/data/harrier-27b-de-lora")
        self.assertEqual(
            paths["distributed_args_path"], "/mnt/data/harrier_de_train_args.json"
        )

    def test_triplet_record_keeps_language_metadata(self):
        record = make_triplet_record(
            query_row={"text": "Behauptung", "pubkey": "p1"},
            positive_text="positive",
            negative_text="negative",
            negative_pubkey="p2",
            negative_rank=7,
            language="de",
            query_key="de_train_0",
        )

        self.assertEqual(record["language"], "de")
        self.assertEqual(record["query_key"], "de_train_0")
        self.assertEqual(record["anchor"], "Behauptung")

    def test_evaluator_payload_uses_language_in_query_ids(self):
        corpus, queries, relevant_docs = build_ir_evaluator_payload(
            [{"pubkey": "p1", "title": "Title", "abstract": "Abstract"}],
            [{"text": "claim", "pubkey": "p1"}],
            language="de",
        )

        self.assertEqual(corpus, {"p1": "Title\nAbstract"})
        self.assertEqual(queries, {"de_dev_0": "claim"})
        self.assertEqual(relevant_docs, {"de_dev_0": {"p1"}})

    def test_build_training_kwargs_defaults_to_german_four_gpu_safe_batching(self):
        kwargs = build_training_kwargs(language="german", data_dir="/data")

        self.assertEqual(kwargs["language"], "de")
        self.assertEqual(kwargs["per_device_train_batch_size"], 1)
        self.assertEqual(kwargs["gradient_accumulation_steps"], 16)
        self.assertEqual(
            kwargs["triplets_path"], "/data/harrier_de_hard_negatives.jsonl"
        )

    def test_resolve_resume_checkpoint_still_uses_latest_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "checkpoint-10").mkdir()
            Path(tmpdir, "checkpoint-20").mkdir()
            Path(tmpdir, "checkpoint-10", "trainer_state.json").write_text("{}")
            Path(tmpdir, "checkpoint-20", "trainer_state.json").write_text("{}")

            self.assertEqual(
                resolve_resume_checkpoint(tmpdir, "auto"),
                str(Path(tmpdir, "checkpoint-20")),
            )


if __name__ == "__main__":
    unittest.main()
