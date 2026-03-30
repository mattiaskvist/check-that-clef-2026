import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


def test_predict_rows_passes_disable_true_when_not_tty(monkeypatch):
    calls: list[bool] = []

    monkeypatch.setattr(main.sys.stderr, "isatty", lambda: False)
    monkeypatch.setattr(main, "_require_gemini_key", lambda: None)
    monkeypatch.setattr(main, "_load_cached_index", lambda _cfg: ([[1.0, 0.0]], [{"pubkey": "1"}]))
    monkeypatch.setattr(main, "load_language_split", lambda _lang, _split: [{"text": "x", "pubkey": 1, "index": 0}])

    class FakeService:
        def __init__(self, *_args, **_kwargs):
            pass
        def embed_texts(self, texts):
            return [[1.0, 0.0] for _ in texts]

    def fake_tqdm(iterable, **kwargs):
        calls.append(bool(kwargs.get("disable")))
        return iterable

    monkeypatch.setattr(main, "tqdm", fake_tqdm)
    monkeypatch.setattr("clef_retrieval.gemini_client.GeminiService", FakeService)
    monkeypatch.setattr(
        main,
        "build_query_embedding_text",
        lambda *_args, **_kwargs: "query",
    )
    monkeypatch.setattr(
        main,
        "rank_from_query_embedding",
        lambda **_kwargs: ["1", "1", "1", "1", "1"],
    )

    rows = main._predict_rows(main.RetrievalConfig(), "en", "dev", 1, None, False)
    assert len(rows) == 1
    assert calls == [True]
