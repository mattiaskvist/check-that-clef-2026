from clef_retrieval.query_policy import select_seeded_subset_indices


def test_seeded_subset_is_reproducible_with_same_seed():
    first = select_seeded_subset_indices(total_count=50, limit=10, seed=42)
    second = select_seeded_subset_indices(total_count=50, limit=10, seed=42)

    assert first == second
    assert len(first) == 10


def test_seeded_subset_changes_with_different_seed():
    first = select_seeded_subset_indices(total_count=50, limit=10, seed=42)
    second = select_seeded_subset_indices(total_count=50, limit=10, seed=43)

    assert first != second
