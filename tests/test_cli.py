import subprocess
import sys
from pathlib import Path

from clef_retrieval.config import RetrievalConfig as RetrievalConfigModel


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "main.py"), *args]
    return subprocess.run(command, check=False, capture_output=True, text=True)


def test_cli_help_lists_required_subcommands():
    result = _run_cli("--help")

    assert result.returncode == 0
    output = result.stdout
    assert "build-index" in output
    assert "predict" in output
    assert "evaluate" in output


def test_predict_requires_lang_and_split_flags():
    result = _run_cli("predict")

    assert result.returncode != 0
    assert "--lang" in result.stderr
    assert "--split" in result.stderr


def test_evaluate_requires_lang_and_split_flags():
    result = _run_cli("evaluate")

    assert result.returncode != 0
    assert "--lang" in result.stderr
    assert "--split" in result.stderr


def test_predict_rejects_non_positive_limit():
    result = _run_cli("predict", "--lang", "en", "--split", "dev", "--limit", "0")

    assert result.returncode != 0
    assert "must be >= 1" in result.stderr


def test_main_returns_2_when_gemini_key_missing(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import main

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        main,
        "index_paths",
        lambda _cfg: (Path("/tmp/nonexistent_embeddings.npy"), Path("/tmp/nonexistent_metadata.jsonl")),
    )
    result = main.main(["build-index"])
    assert result == 2


def test_main_returns_1_when_cache_missing(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import main

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(main, "RetrievalConfig", lambda: RetrievalConfigModel(cache_dir=str(tmp_path)))
    result = main.main(["predict", "--lang", "en", "--split", "dev", "--limit", "1"])
    assert result == 1


def test_build_index_accepts_limit_papers_argument():
    result = _run_cli("build-index", "--limit-papers", "10", "--help")
    assert result.returncode == 0


def test_build_index_rejects_non_positive_limit_papers():
    result = _run_cli("build-index", "--limit-papers", "0")
    assert result.returncode != 0
    assert "must be >= 1" in result.stderr


def test_predict_prints_summary_when_successful(monkeypatch, tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import main

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(main, "RetrievalConfig", lambda: RetrievalConfigModel(cache_dir=str(tmp_path)))
    monkeypatch.setattr(main, "_predict_rows", lambda *_args, **_kwargs: [{"top5": ["1"] * 5} for _ in range(3)])
    monkeypatch.setattr(main, "_write_predictions", lambda *_args, **_kwargs: None)

    exit_code = main.main(["predict", "--lang", "en", "--split", "dev", "--limit", "3"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Predict summary:" in captured


def test_predict_accepts_query_batch_size_argument():
    result = _run_cli("predict", "--lang", "en", "--split", "dev", "--query-batch-size", "16", "--help")
    assert result.returncode == 0


def test_predict_accepts_skip_query_extraction_argument():
    result = _run_cli("predict", "--lang", "en", "--split", "dev", "--skip-query-extraction", "--help")
    assert result.returncode == 0


def test_evaluate_uses_cached_predictions_without_recomputing(monkeypatch, tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import main

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = cache_dir / "predictions_en_dev.jsonl"
    prediction_path.write_text(
        '\n'.join(
            [
                '{"index": 0, "text": "t0", "pubkey": "1", "top5": ["1", "2", "3", "4", "5"]}',
                '{"index": 1, "text": "t1", "pubkey": "2", "top5": ["2", "1", "3", "4", "5"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "RetrievalConfig", lambda: RetrievalConfigModel(cache_dir=str(cache_dir)))
    monkeypatch.setattr(main, "_predict_rows", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not recompute predictions")))
    monkeypatch.setattr(main, "scorer", lambda top5_preds, lang, split: 0.5)

    exit_code = main.main(["evaluate", "--lang", "en", "--split", "dev"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Using cached predictions from" in output
    assert "MRR@5:" in output


def test_evaluate_accepts_recompute_argument():
    result = _run_cli("evaluate", "--lang", "en", "--split", "dev", "--recompute", "--help")
    assert result.returncode == 0


def test_predict_accepts_subset_seed_argument():
    result = _run_cli("predict", "--lang", "en", "--split", "dev", "--subset-seed", "123", "--help")
    assert result.returncode == 0


def test_predict_accepts_subset_per_language_limit_argument():
    result = _run_cli(
        "predict",
        "--lang",
        "en",
        "--split",
        "dev",
        "--subset-per-language-limit",
        "50",
        "--help",
    )
    assert result.returncode == 0


def test_evaluate_accepts_multilingual_metrics_argument():
    result = _run_cli("evaluate", "--lang", "en", "--split", "dev", "--multilingual-metrics", "--help")
    assert result.returncode == 0
