import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


def test_usage_estimate_counts_model_calls():
    usage = main._estimate_query_api_usage(query_count=10, skip_query_extraction=False)

    assert usage["extraction_mode"] == "enabled"
    assert usage["calls_per_query"] == 2
    assert usage["extraction_calls"] == 10
    assert usage["embedding_calls"] == 10
    assert usage["total_calls"] == 20


def test_cost_guardrail_raises_when_limit_exceeded():
    try:
        main._enforce_cost_guardrail(
            total_calls=20,
            max_calls=10,
            allow_exceed=False,
        )
    except ValueError as exc:
        assert "Cost guardrail exceeded" in str(exc)
    else:
        raise AssertionError("Expected cost guardrail to raise ValueError")


def test_predict_reports_usage_and_guardrail_warning(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    config_cls = main.RetrievalConfig
    monkeypatch.setattr(main, "RetrievalConfig", lambda: config_cls.model_construct(cache_dir=str(tmp_path)))
    monkeypatch.setattr(
        main,
        "_predict_rows",
        lambda *_args, **_kwargs: (
            [{"top5": ["1", "1", "1", "1", "1"]} for _ in range(3)],
            {"parsed+accepted": 2, "parsed+rejected": 1, "error->fallback": 0},
        ),
    )
    monkeypatch.setattr(main, "_write_predictions", lambda *_args, **_kwargs: None)

    exit_code = main.main(
        [
            "predict",
            "--lang",
            "en",
            "--split",
            "dev",
            "--limit",
            "3",
            "--max-estimated-api-calls",
            "2",
            "--allow-cost-overrun",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Usage estimate:" in output
    assert "projected_cost_sek=" in output
    assert "Guardrail status: WARNING" in output
    assert "Extraction outcomes:" in output
