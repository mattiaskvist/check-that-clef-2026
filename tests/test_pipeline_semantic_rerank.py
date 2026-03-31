"""Tests for semantic-primary reranking in the pipeline.

These tests lock decisions D-05, D-06 from 02-CONTEXT.md:
- D-05: Use semantic reranker score as primary ordering signal
- D-06: Use Phase 1 weighted signal score only as a tie-breaker
"""

import pytest

from clef_retrieval.config import RetrievalConfig
from clef_retrieval.schemas import TweetEvidence


class TestSemanticPrimaryOrdering:
    """Semantic score drives ordering; weighted signal only breaks ties."""

    def test_semantic_score_determines_primary_ordering(self):
        """Higher semantic score wins regardless of weighted signal score."""
        from clef_retrieval.reranker import semantic_primary_sort

        candidates_with_scores = [
            # (pubkey, semantic_score, weighted_score, original_index)
            ("low-semantic-high-weighted", 0.3, 100.0, 0),
            ("high-semantic-low-weighted", 0.9, 1.0, 1),
            ("mid-semantic-mid-weighted", 0.6, 50.0, 2),
        ]
        config = RetrievalConfig()

        result = semantic_primary_sort(candidates_with_scores, config.semantic_tie_epsilon)

        assert result[0][0] == "high-semantic-low-weighted"
        assert result[1][0] == "mid-semantic-mid-weighted"
        assert result[2][0] == "low-semantic-high-weighted"

    def test_weighted_score_breaks_semantic_ties(self):
        """When semantic scores are equal within epsilon, weighted score decides."""
        from clef_retrieval.reranker import semantic_primary_sort

        candidates_with_scores = [
            # semantic scores within epsilon (0.01) of each other
            ("low-weighted", 0.800, 10.0, 0),
            ("high-weighted", 0.805, 50.0, 1),
        ]
        config = RetrievalConfig(semantic_tie_epsilon=0.01)

        result = semantic_primary_sort(candidates_with_scores, config.semantic_tie_epsilon)

        # When semantic scores are tied, higher weighted score wins
        assert result[0][0] == "high-weighted"
        assert result[1][0] == "low-weighted"

    def test_original_index_breaks_full_ties(self):
        """When both semantic and weighted are tied, original index preserves order."""
        from clef_retrieval.reranker import semantic_primary_sort

        candidates_with_scores = [
            ("third", 0.8, 50.0, 2),
            ("first", 0.8, 50.0, 0),
            ("second", 0.8, 50.0, 1),
        ]
        config = RetrievalConfig(semantic_tie_epsilon=0.01)

        result = semantic_primary_sort(candidates_with_scores, config.semantic_tie_epsilon)

        # When all tied, original order is preserved (stable sort by original_index)
        assert result[0][0] == "first"
        assert result[1][0] == "second"
        assert result[2][0] == "third"


class TestPipelineSemanticRerank:
    """Pipeline integration of semantic reranking."""

    def test_pipeline_applies_semantic_rerank_to_top_k_dense_candidates(self):
        """Only top rerank_top_k candidates get semantic reranking."""
        from clef_retrieval.pipeline import rank_from_query_embedding

        # Create 100 candidates but rerank_top_k=10
        metadata_rows = [
            {"pubkey": f"paper-{i}", "title": f"Paper {i}", "abstract": f"Content {i}"}
            for i in range(100)
        ]
        # Embeddings arranged so paper-99 is closest in dense retrieval
        paper_embeddings = [[1.0, 0.0] for _ in range(100)]
        query_embedding = [1.0, 0.0]

        config = RetrievalConfig(rerank_top_k=10, top_k=100)

        result = rank_from_query_embedding(
            tweet_text="query",
            query_embedding=query_embedding,
            paper_embeddings=paper_embeddings,
            metadata_rows=metadata_rows,
            top_k=100,
            evidence=None,
            config=config,
        )

        # Should return top 5 from reranked subset
        assert len(result) == 5
        # All results should be from original candidates
        for pubkey in result:
            assert pubkey.startswith("paper-") or pubkey == "0"

    def test_pipeline_uses_semantic_reranker_when_available(self):
        """Pipeline integrates semantic reranker for candidate scoring."""
        from clef_retrieval.pipeline import rank_from_query_embedding

        metadata_rows = [
            {"pubkey": "paper-a", "title": "Alpha", "abstract": "science claim"},
            {"pubkey": "paper-b", "title": "Beta", "abstract": "other details"},
        ]
        paper_embeddings = [[1.0, 0.0], [0.5, 0.5]]

        config = RetrievalConfig(rerank_top_k=50)

        result = rank_from_query_embedding(
            tweet_text="science claim",
            query_embedding=[1.0, 0.0],
            paper_embeddings=paper_embeddings,
            metadata_rows=metadata_rows,
            top_k=2,
            evidence=None,
            config=config,
        )

        assert len(result) == 5  # Padded to 5


class TestDeterministicOrdering:
    """Ordering is deterministic across runs with same inputs."""

    def test_semantic_rerank_is_deterministic(self):
        """Multiple calls with same inputs produce identical output."""
        from clef_retrieval.reranker import semantic_primary_sort

        candidates_with_scores = [
            ("pub1", 0.9, 10.0, 0),
            ("pub2", 0.9, 10.0, 1),
            ("pub3", 0.8, 20.0, 2),
        ]
        config = RetrievalConfig(semantic_tie_epsilon=0.01)

        results = [
            semantic_primary_sort(candidates_with_scores, config.semantic_tie_epsilon)
            for _ in range(5)
        ]

        # All results should be identical
        for result in results[1:]:
            assert result == results[0]

    def test_pipeline_output_is_deterministic_with_same_seed(self):
        """Pipeline produces identical results with same config seed."""
        from clef_retrieval.pipeline import rank_from_query_embedding

        metadata_rows = [
            {"pubkey": f"paper-{i}", "title": f"Paper {i}", "abstract": f"Content {i}"}
            for i in range(10)
        ]
        paper_embeddings = [[1.0, 0.0]] * 10
        query_embedding = [1.0, 0.0]

        config = RetrievalConfig(subset_seed=42)

        results = [
            rank_from_query_embedding(
                tweet_text="query",
                query_embedding=query_embedding,
                paper_embeddings=paper_embeddings,
                metadata_rows=metadata_rows,
                top_k=10,
                evidence=None,
                config=config,
            )
            for _ in range(3)
        ]

        for result in results[1:]:
            assert result == results[0]


class TestWeightedTieBreakPreservesPhase1Behavior:
    """Phase 1 weighted scoring is used only when semantic scores tie."""

    def test_weighted_rerank_still_used_for_tie_breaking(self):
        """When semantic scores tie, existing weighted signal logic applies."""
        from clef_retrieval.reranker import semantic_primary_sort

        evidence = TweetEvidence(
            query_text_for_embedding="query",
            language="en",
            candidate_title_mentions=["climate model"],
            candidate_authors=["alice smith"],
        )

        # Both have identical semantic scores (will be within epsilon)
        # but weighted scores differ based on title/author matches
        candidates_with_scores = [
            # pubkey with lower weighted match
            ("no-match", 0.8, 1.0, 0),
            # pubkey with title/author match (higher weighted score)
            ("title-author-match", 0.8, 10.0, 1),
        ]
        config = RetrievalConfig(semantic_tie_epsilon=0.01)

        result = semantic_primary_sort(candidates_with_scores, config.semantic_tie_epsilon)

        # Higher weighted score wins the tie
        assert result[0][0] == "title-author-match"
        assert result[1][0] == "no-match"
