import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


def test_select_subset_rows_is_seeded_and_reproducible():
    rows = [{"index": idx, "text": f"t{idx}", "pubkey": str(idx)} for idx in range(12)]

    first_rows, first_indices = main._select_subset_rows(rows, subset_limit=5, subset_seed=7)
    second_rows, second_indices = main._select_subset_rows(rows, subset_limit=5, subset_seed=7)
    third_rows, third_indices = main._select_subset_rows(rows, subset_limit=5, subset_seed=8)

    assert first_indices == second_indices
    assert [row["index"] for row in first_rows] == [row["index"] for row in second_rows]
    assert first_indices != third_indices


def test_evaluate_multilingual_metrics_outputs_per_language_and_overall(monkeypatch, tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for lang in ("en", "de", "fr"):
        (cache_dir / f"predictions_{lang}_dev.jsonl").write_text(
            '{"index": 0, "text": "tweet", "pubkey": "1", "top5": ["1", "2", "3", "4", "5"]}\n',
            encoding="utf-8",
        )

    config_cls = main.RetrievalConfig
    monkeypatch.setattr(main, "RetrievalConfig", lambda: config_cls.model_construct(cache_dir=str(cache_dir)))
    monkeypatch.setattr(
        main,
        "scorer",
        lambda _preds, lang, split: {"en": 0.10, "de": 0.20, "fr": 0.30}[lang],
    )

    exit_code = main.main(["evaluate", "--lang", "en", "--split", "dev", "--multilingual-metrics"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "MRR@5 [en]:" in output
    assert "MRR@5 [de]:" in output
    assert "MRR@5 [fr]:" in output
    assert "MRR@5 overall (en,de,fr):" in output


def test_evaluate_promotion_gate_blocks_regression(monkeypatch, tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for lang in ("en", "de", "fr"):
        (cache_dir / f"predictions_{lang}_dev.jsonl").write_text(
            '{"index": 0, "text": "tweet", "pubkey": "1", "top5": ["1", "2", "3", "4", "5"]}\n',
            encoding="utf-8",
        )

    config_cls = main.RetrievalConfig
    monkeypatch.setattr(main, "RetrievalConfig", lambda: config_cls.model_construct(cache_dir=str(cache_dir)))
    monkeypatch.setattr(
        main,
        "scorer",
        lambda _preds, lang, split: {"en": 0.31, "de": 0.24, "fr": 0.32}[lang],
    )

    exit_code = main.main(
        [
            "evaluate",
            "--lang",
            "en",
            "--split",
            "dev",
            "--multilingual-metrics",
            "--promotion-baseline-overall",
            "0.25",
            "--promotion-baseline-en",
            "0.30",
            "--promotion-baseline-de",
            "0.25",
            "--promotion-baseline-fr",
            "0.30",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Promotion gate: BLOCKED" in output
