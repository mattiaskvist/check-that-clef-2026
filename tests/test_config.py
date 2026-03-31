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
    assert cfg.query_batch_size == 32
    assert cfg.extraction_gate_min_title_mentions == 1
    assert cfg.extraction_gate_min_authors == 1
    assert cfg.extraction_gate_min_filled_key_fields == 2
    assert cfg.weight_title_author == 5.0
    assert cfg.weight_method_finding == 3.0
    assert cfg.weight_keywords == 1.0
    assert cfg.language_normalization_mode == "strict"
    assert cfg.subset_seed == 42
    assert cfg.subset_per_language_limit == 100


def test_weight_order_validation_rejects_invalid_order():
    try:
        RetrievalConfig(
            weight_title_author=2.0,
            weight_method_finding=3.0,
            weight_keywords=1.0,
        )
        assert False, "Expected ValueError for invalid weight ordering"
    except ValueError as exc:
        assert "weight ordering" in str(exc)
