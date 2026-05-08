import unittest
import tempfile
from pathlib import Path

from clef_pipeline.retrievers import HarrierRetriever
from clef_training.train_harrier_en_modal import (
    article_to_text,
    build_ir_evaluator_payload,
    find_latest_checkpoint,
    load_completed_query_keys,
    make_triplet_record,
    parse_target_modules,
    resolve_resume_checkpoint,
    texts_fingerprint,
)


class HarrierEnglishTrainingTests(unittest.TestCase):
    def test_article_to_text_uses_title_and_abstract(self):
        doc = {"title": "Clinical trial", "abstract": "Findings here."}

        self.assertEqual(article_to_text(doc), "Clinical trial\nFindings here.")

    def test_parse_target_modules_accepts_all_linear(self):
        self.assertEqual(parse_target_modules("all-linear"), "all-linear")

    def test_parse_target_modules_accepts_comma_list(self):
        self.assertEqual(parse_target_modules("q_proj, v_proj"), ["q_proj", "v_proj"])

    def test_make_triplet_record_keeps_audit_metadata(self):
        record = make_triplet_record(
            query_row={"text": "claim text", "pubkey": "p1"},
            positive_text="positive paper",
            negative_text="negative paper",
            negative_pubkey="p2",
            negative_rank=3,
        )

        self.assertEqual(record["anchor"], "claim text")
        self.assertEqual(record["query_key"], "p1")
        self.assertEqual(record["positive"], "positive paper")
        self.assertEqual(record["negative"], "negative paper")
        self.assertEqual(record["query_pubkey"], "p1")
        self.assertEqual(record["negative_pubkey"], "p2")
        self.assertEqual(record["negative_rank"], 3)
        self.assertEqual(record["language"], "en")

    def test_build_ir_evaluator_payload_maps_dev_labels(self):
        corpus, queries, relevant_docs = build_ir_evaluator_payload(
            [{"pubkey": "p1", "title": "Title", "abstract": "Abstract"}],
            [{"text": "claim", "pubkey": "p1"}],
        )

        self.assertEqual(corpus, {"p1": "Title\nAbstract"})
        self.assertEqual(queries, {"en_dev_0": "claim"})
        self.assertEqual(relevant_docs, {"en_dev_0": {"p1"}})

    def test_texts_fingerprint_depends_on_text_content(self):
        self.assertEqual(texts_fingerprint(["abc"]), texts_fingerprint(["abc"]))
        self.assertNotEqual(texts_fingerprint(["abc"]), texts_fingerprint(["abcd"]))

    def test_load_completed_query_keys_skips_bad_jsonl_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, "triplets.jsonl")
            path.write_text(
                '{"query_key": "en_train_0"}\nnot-json\n{"query_pubkey": "p2"}\n'
            )

            self.assertEqual(load_completed_query_keys(str(path)), {"en_train_0", "p2"})

    def test_harrier_cache_key_includes_lora_adapter(self):
        retriever = HarrierRetriever.__new__(HarrierRetriever)
        retriever.model_name = "microsoft/harrier-oss-v1-27b"
        retriever.lora_id = "example/adapter"

        self.assertIn("example--adapter", retriever._cache_key())

    def test_find_latest_checkpoint_uses_highest_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "checkpoint-10").mkdir()
            Path(tmpdir, "checkpoint-250").mkdir()
            Path(tmpdir, "checkpoint-10", "trainer_state.json").write_text("{}")
            Path(tmpdir, "checkpoint-250", "trainer_state.json").write_text("{}")
            Path(tmpdir, "checkpoint-999").mkdir()
            Path(tmpdir, "checkpoint-final").mkdir()

            self.assertEqual(
                find_latest_checkpoint(tmpdir),
                str(Path(tmpdir, "checkpoint-250")),
            )

    def test_resolve_resume_checkpoint_auto_or_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "checkpoint-42").mkdir()
            Path(tmpdir, "checkpoint-42", "trainer_state.json").write_text("{}")

            self.assertEqual(
                resolve_resume_checkpoint(tmpdir, "auto"),
                str(Path(tmpdir, "checkpoint-42")),
            )
            self.assertIsNone(resolve_resume_checkpoint(tmpdir, "none"))
            self.assertEqual(
                resolve_resume_checkpoint(tmpdir, "/custom/checkpoint"),
                "/custom/checkpoint",
            )


if __name__ == "__main__":
    unittest.main()
