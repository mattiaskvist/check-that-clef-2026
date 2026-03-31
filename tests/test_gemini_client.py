from clef_retrieval.gemini_client import build_embedding_input
from clef_retrieval.schemas import TweetEvidence


def test_build_embedding_input_prefers_evidence_text():
    evidence = TweetEvidence(query_text_for_embedding="structured query")

    value = build_embedding_input(evidence, "raw tweet text")

    assert value == "structured query"


def test_build_embedding_input_falls_back_to_raw_tweet_when_empty():
    evidence = TweetEvidence(query_text_for_embedding="")

    value = build_embedding_input(evidence, "raw tweet text")

    assert value == "raw tweet text"


class _FakeEmbedding:
    def __init__(self, values: list[float]):
        self.values = values


class _FakeEmbedResponse:
    def __init__(self, embeddings: list[_FakeEmbedding]):
        self.embeddings = embeddings


class _FakeModels:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.failures_before_success = 0

    def embed_content(self, *, model: str, contents: list[str]):
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            from google.genai import errors

            raise errors.APIError(
                429,
                {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota"}},
            )
        self.calls.append(contents)
        embeddings = [_FakeEmbedding([float(i)]) for i, _ in enumerate(contents)]
        return _FakeEmbedResponse(embeddings)


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


def test_embed_texts_batches_requests_in_chunks_of_100():
    from clef_retrieval.config import RetrievalConfig
    from clef_retrieval.gemini_client import GeminiService

    fake_client = _FakeClient()
    service = GeminiService(config=RetrievalConfig(), client=fake_client)

    texts = [f"text-{i}" for i in range(205)]
    result = service.embed_texts(texts)

    assert len(result) == 205
    assert [len(call) for call in fake_client.models.calls] == [100, 100, 5]


def test_embed_texts_retries_on_quota_exhaustion(monkeypatch):
    from clef_retrieval.config import RetrievalConfig
    import clef_retrieval.gemini_client as gemini_client
    from clef_retrieval.gemini_client import GeminiService

    fake_client = _FakeClient()
    fake_client.models.failures_before_success = 2
    sleep_calls: list[float] = []
    monkeypatch.setattr(gemini_client.time, "sleep", lambda sec: sleep_calls.append(sec))

    cfg = RetrievalConfig(embed_max_retries=5, embed_backoff_base_seconds=1.0)
    service = GeminiService(config=cfg, client=fake_client)
    result = service.embed_texts(["a", "b"])

    assert len(result) == 2
    assert len(fake_client.models.calls) == 1
    assert sleep_calls == [1.0, 2.0, cfg.embed_min_interval_seconds]


def test_embed_texts_progress_bar_disabled_when_not_tty(monkeypatch):
    from clef_retrieval.config import RetrievalConfig
    import clef_retrieval.gemini_client as gemini_client
    from clef_retrieval.gemini_client import GeminiService

    fake_client = _FakeClient()
    calls: list[bool] = []

    monkeypatch.setattr(gemini_client.sys.stderr, "isatty", lambda: False)

    def fake_tqdm(iterable, **kwargs):
        calls.append(bool(kwargs.get("disable")))
        return iterable

    monkeypatch.setattr(gemini_client, "tqdm", fake_tqdm)
    monkeypatch.setattr(gemini_client.time, "sleep", lambda _sec: None)

    service = GeminiService(config=RetrievalConfig(embed_batch_size=2), client=fake_client)
    _ = service.embed_texts(["a", "b", "c"])

    assert calls == [True]
