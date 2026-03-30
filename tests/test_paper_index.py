import numpy as np

from clef_retrieval.config import RetrievalConfig
from clef_retrieval.paper_index import (
    IndexCacheError,
    build_or_load_index,
    build_paper_text,
    validate_cached_index,
)
from clef_retrieval.schemas import PaperEvidence


class FakeGeminiService:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


def test_build_paper_text_includes_core_fields_and_optional_sections():
    evidence = PaperEvidence(
        pubkey="p1",
        title="Paper title",
        authors="Alice; Bob",
        abstract="Important abstract",
        venue="Journal X",
        keywords=["kw1", "kw2"],
        highlights=["h1", "h2"],
        paper_text_for_embedding="ignored for this builder",
    )

    text = build_paper_text(evidence)

    assert "Title: Paper title" in text
    assert "Authors: Alice; Bob" in text
    assert "Abstract: Important abstract" in text
    assert "Venue: Journal X" in text
    assert "Keywords: kw1, kw2" in text
    assert "Highlights: h1; h2" in text


def test_build_or_load_index_writes_and_reuses_cache(tmp_path):
    config = RetrievalConfig(cache_dir=str(tmp_path / "cache"))
    collection = [
        {
            "pubkey": 1,
            "title": "First paper",
            "authors": ["Alice", "Bob"],
            "abstract": "First abstract",
            "venue": "Venue A",
        },
        {
            "pubkey": 2,
            "title": "Second paper",
            "authors": "Carol",
            "abstract": "Second abstract",
            "venue": "Venue B",
        },
    ]

    built = build_or_load_index(collection, FakeGeminiService(), config, force_rebuild=True)
    loaded = build_or_load_index(collection, FakeGeminiService(), config, force_rebuild=False)

    assert built["built"] is True
    assert loaded["built"] is False
    assert built["count"] == 2
    assert loaded["count"] == 2
    assert built["metadata_path"].endswith("paper_metadata.jsonl")
    assert built["embeddings_path"].endswith("paper_embeddings.npy")
    assert np.array_equal(loaded["embeddings"], built["embeddings"])


def test_validate_cached_index_rejects_length_mismatch():
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
    metadata = [{"pubkey": "1"}]

    try:
        validate_cached_index(embeddings, metadata)
        assert False, "Expected IndexCacheError for mismatched cache lengths"
    except IndexCacheError as error:
        assert "does not match metadata rows" in str(error)
