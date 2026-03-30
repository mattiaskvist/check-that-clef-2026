from clef_retrieval.data import load_collection, load_language_split, normalize_authors


def test_normalize_authors_handles_lists_and_strings():
    assert normalize_authors(["A. One", "B. Two"]) == "A. One; B. Two"
    assert normalize_authors(" C. Three ") == "C. Three"
    assert normalize_authors(None) == ""


def test_load_collection_uses_collection_subset(monkeypatch):
    calls = {}

    def fake_load_dataset(dataset_id, subset):
        calls["args"] = (dataset_id, subset)
        return {"collection": ["paper-1", "paper-2"]}

    monkeypatch.setattr("clef_retrieval.data.load_dataset", fake_load_dataset)

    assert load_collection() == ["paper-1", "paper-2"]
    assert calls["args"] == (
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        "collection",
    )


def test_load_language_split_uses_requested_language_and_split(monkeypatch):
    calls = {}

    def fake_load_dataset(dataset_id, subset):
        calls["args"] = (dataset_id, subset)
        return {"train": ["tweet-1"], "test": ["tweet-2"]}

    monkeypatch.setattr("clef_retrieval.data.load_dataset", fake_load_dataset)

    assert load_language_split("en", "test") == ["tweet-2"]
    assert calls["args"] == (
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        "en",
    )
