from clef_retrieval.config import RetrievalConfig


def test_default_config_values():
    cfg = RetrievalConfig()
    assert cfg.embedding_model == "models/gemini-embedding-2-preview"
    assert cfg.rerank_model == "gemini-3.1-flash-lite-preview"
    assert cfg.query_model == "gemini-3.1-flash-lite-preview"
    assert cfg.top_k == 200
    assert cfg.top_n == 5
    assert cfg.cache_dir == ".cache/clef_retrieval"
    assert cfg.embed_batch_size == 100
    assert cfg.embed_min_interval_seconds == 1.1
    assert cfg.embed_max_retries == 5
    assert cfg.embed_backoff_base_seconds == 1.5
