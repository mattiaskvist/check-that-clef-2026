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
