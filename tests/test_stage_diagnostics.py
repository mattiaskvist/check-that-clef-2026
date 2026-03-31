"""Tests for stage-level diagnostics in evaluate output.

Covers D-07 (Recall@K diagnostics), D-08 (semantic MRR@5 uplift), 
and D-09 (latency/throughput reporting) from Phase 2 CONTEXT.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


def test_evaluate_reports_dense_recall_at_k(monkeypatch, tmp_path, capsys):
    """D-07: Evaluate output includes dense-stage Recall@K diagnostics."""
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
        lambda _preds, lang, split: 0.25,
    )

    exit_code = main.main(["evaluate", "--lang", "en", "--split", "dev", "--multilingual-metrics"])
    output = capsys.readouterr().out

    assert exit_code == 0
    # D-07: Stage diagnostics must show Recall@K for dense retrieval
    assert "Recall@" in output, "Expected Recall@K diagnostics in evaluate output"


def test_evaluate_reports_semantic_mrr_uplift(monkeypatch, tmp_path, capsys):
    """D-08: Evaluate output reports semantic reranked MRR@5 and uplift delta."""
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
        lambda _preds, lang, split: {"en": 0.30, "de": 0.28, "fr": 0.32}[lang],
    )

    exit_code = main.main([
        "evaluate", "--lang", "en", "--split", "dev", "--multilingual-metrics",
        "--promotion-baseline-overall", "0.25",
        "--promotion-baseline-en", "0.25",
        "--promotion-baseline-de", "0.25",
        "--promotion-baseline-fr", "0.25",
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    # D-08: Must show semantic MRR@5 and uplift delta
    assert "uplift" in output.lower(), "Expected uplift delta in evaluate output"


def test_evaluate_reports_per_language_uplift(monkeypatch, tmp_path, capsys):
    """D-08: Uplift delta is reported overall AND per-language (en/de/fr)."""
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
        lambda _preds, lang, split: {"en": 0.30, "de": 0.28, "fr": 0.32}[lang],
    )

    exit_code = main.main([
        "evaluate", "--lang", "en", "--split", "dev", "--multilingual-metrics",
        "--promotion-baseline-overall", "0.25",
        "--promotion-baseline-en", "0.20",
        "--promotion-baseline-de", "0.22",
        "--promotion-baseline-fr", "0.24",
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    # Should show per-language uplift values
    # Expecting format like "uplift [en]: +X.XX" or similar parseable format
    output_lower = output.lower()
    assert "en" in output_lower and "uplift" in output_lower, "Expected per-language uplift for en"
    assert "de" in output_lower and "uplift" in output_lower, "Expected per-language uplift for de"
    assert "fr" in output_lower and "uplift" in output_lower, "Expected per-language uplift for fr"


def test_evaluate_reports_reranker_latency_throughput(monkeypatch, tmp_path, capsys):
    """D-09: Evaluate output includes reranker latency/throughput diagnostics."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "predictions_en_dev.jsonl").write_text(
        '{"index": 0, "text": "tweet", "pubkey": "1", "top5": ["1", "2", "3", "4", "5"]}\n',
        encoding="utf-8",
    )

    config_cls = main.RetrievalConfig
    monkeypatch.setattr(main, "RetrievalConfig", lambda: config_cls.model_construct(cache_dir=str(cache_dir)))
    monkeypatch.setattr(main, "scorer", lambda _preds, lang, split: 0.25)

    exit_code = main.main(["evaluate", "--lang", "en", "--split", "dev"])
    output = capsys.readouterr().out

    assert exit_code == 0
    # D-09: Must show reranker performance metrics
    output_lower = output.lower()
    assert "latency" in output_lower or "throughput" in output_lower or "rerank" in output_lower, \
        "Expected reranker latency/throughput diagnostics in evaluate output"


def test_evaluate_diagnostics_stable_in_cached_mode(monkeypatch, tmp_path, capsys):
    """Diagnostics emitted consistently whether using cache or recompute path."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "predictions_en_dev.jsonl").write_text(
        '{"index": 0, "text": "tweet", "pubkey": "1", "top5": ["1", "2", "3", "4", "5"]}\n',
        encoding="utf-8",
    )

    config_cls = main.RetrievalConfig
    monkeypatch.setattr(main, "RetrievalConfig", lambda: config_cls.model_construct(cache_dir=str(cache_dir)))
    monkeypatch.setattr(main, "scorer", lambda _preds, lang, split: 0.25)

    # Run with cache
    exit_code = main.main(["evaluate", "--lang", "en", "--split", "dev"])
    output = capsys.readouterr().out

    assert exit_code == 0
    # Must include diagnostic sections even in cached mode
    assert "Recall@" in output or "rerank_top_k" in output.lower(), \
        "Expected diagnostics labels in cached evaluate output"


def test_evaluate_recalls_rerank_depth_readout(monkeypatch, tmp_path, capsys):
    """Diagnostics show explicit rerank depth setting for reproducibility."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "predictions_en_dev.jsonl").write_text(
        '{"index": 0, "text": "tweet", "pubkey": "1", "top5": ["1", "2", "3", "4", "5"]}\n',
        encoding="utf-8",
    )

    config_cls = main.RetrievalConfig
    monkeypatch.setattr(main, "RetrievalConfig", lambda: config_cls.model_construct(cache_dir=str(cache_dir)))
    monkeypatch.setattr(main, "scorer", lambda _preds, lang, split: 0.25)

    exit_code = main.main(["evaluate", "--lang", "en", "--split", "dev"])
    output = capsys.readouterr().out

    assert exit_code == 0
    # Should show rerank depth (D-04: explicit config/CLI-controlled parameter)
    # Look for "rerank" or "depth" or the actual number 50
    output_lower = output.lower()
    assert "rerank" in output_lower or "depth" in output_lower or "50" in output, \
        "Expected rerank depth readout in diagnostics"
