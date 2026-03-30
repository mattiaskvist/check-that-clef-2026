import numpy as np

from clef_retrieval.retriever import l2_normalize, retrieve_top_pubkeys, topk_indices


def test_l2_normalize_scales_rows_to_unit_norm():
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]])

    normalized = l2_normalize(matrix)

    assert np.isclose(np.linalg.norm(normalized[0]), 1.0)
    assert np.array_equal(normalized[1], np.array([0.0, 0.0]))


def test_topk_indices_returns_descending_similarity():
    sims = np.array([0.2, 0.9, 0.4, 0.8])

    assert topk_indices(sims, k=2) == [1, 3]


def test_retrieve_top_pubkeys_uses_cosine_similarity_ordering():
    query_embedding = np.array([1.0, 0.0])
    paper_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )

    pubkeys = ["paper-a", "paper-b", "paper-c"]

    result = retrieve_top_pubkeys(query_embedding, paper_embeddings, pubkeys, k=2)

    assert result == ["paper-a", "paper-b"]


def test_retrieve_top_pubkeys_rejects_length_mismatch():
    query_embedding = np.array([1.0, 0.0])
    paper_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    try:
        retrieve_top_pubkeys(query_embedding, paper_embeddings, ["only-one"], k=1)
        assert False, "Expected ValueError for mismatched embedding/pubkey lengths"
    except ValueError as error:
        assert "Length mismatch" in str(error)
