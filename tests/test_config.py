from clef_retrieval.config import RetrievalConfig


def test_default_config_values():
    cfg = RetrievalConfig()
    assert cfg.embedding_model == "models/gemini-embedding-2-preview"
    assert cfg.rerank_model == "gemini-3.1-flash-lite-preview"
    assert cfg.query_model == "gemini-3.1-flash-lite-preview"
    assert cfg.top_k == 200
    assert cfg.top_n == 5
    assert cfg.cache_dir == ".cache/clef_retrieval"
