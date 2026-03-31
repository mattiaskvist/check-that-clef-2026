"""Tests for duplicate-title disambiguation logic (RNK-03)."""

from clef_retrieval.config import RetrievalConfig
from clef_retrieval.disambiguation import (
    compute_author_similarity,
    compute_disambiguation_score,
    compute_method_match,
    compute_venue_match,
    detect_title_collisions,
    disambiguate_title_collisions,
    normalize_title,
)
from clef_retrieval.schemas import TweetEvidence


def test_normalize_title_handles_case_and_punctuation():
    """D-02: Normalized title matching handles casing and punctuation."""
    assert normalize_title("A Study of Climate Change") == normalize_title("a study of climate change")
    assert normalize_title('"A Study of Climate Change"') == normalize_title("A Study of Climate Change")
    assert normalize_title("A Study of Climate Change.") == normalize_title("A Study of Climate Change")
    assert normalize_title("A Study of Climate Change!") == normalize_title("A Study of Climate Change")


def test_normalize_title_handles_unicode():
    """D-02: Normalized title matching handles Unicode variants."""
    # NFKC normalization should handle composed/decomposed forms
    assert normalize_title("Café") == normalize_title("Café")  # Different Unicode representations


def test_detect_title_collisions_returns_only_collision_groups():
    """D-01: Collision detection only activates when duplicates exist."""
    metadata_by_pubkey = {
        "paper1": {"title": "Climate Model Study"},
        "paper2": {"title": "Climate Model Study"},  # Duplicate
        "paper3": {"title": "Unique Paper Title"},
        "paper4": {"title": "Another Unique Title"},
    }
    
    collision_groups = detect_title_collisions(
        candidates=["paper1", "paper2", "paper3", "paper4"],
        metadata_by_pubkey=metadata_by_pubkey,
    )
    
    # Should only return groups with >1 member (D-01: lazy evaluation)
    assert len(collision_groups) == 1
    normalized_title = normalize_title("Climate Model Study")
    assert normalized_title in collision_groups
    assert set(collision_groups[normalized_title]) == {"paper1", "paper2"}


def test_detect_title_collisions_handles_no_duplicates():
    """D-01: Returns empty dict when no collisions exist."""
    metadata_by_pubkey = {
        "paper1": {"title": "Climate Model Study"},
        "paper2": {"title": "Weather Pattern Analysis"},
        "paper3": {"title": "Ocean Temperature Trends"},
    }
    
    collision_groups = detect_title_collisions(
        candidates=["paper1", "paper2", "paper3"],
        metadata_by_pubkey=metadata_by_pubkey,
    )
    
    # No duplicates, should return empty dict
    assert collision_groups == {}


def test_compute_author_similarity_exact_match():
    """D-03: Author signal has highest disambiguation weight."""
    tweet_authors = ["Alice Smith"]
    paper_authors = "Alice Smith, Bob Jones"
    
    similarity = compute_author_similarity(tweet_authors, paper_authors)
    
    # Exact substring match should return 1.0
    assert similarity == 1.0


def test_compute_author_similarity_fuzzy_match():
    """D-03: Author signal uses fuzzy matching for partial matches."""
    tweet_authors = ["Alice Smith"]
    paper_authors = "A. Smith, Bob Jones"
    
    similarity = compute_author_similarity(tweet_authors, paper_authors)
    
    # Should have high similarity (>0.5) for partial match
    assert similarity > 0.5


def test_compute_author_similarity_no_match():
    """D-03: Author signal returns 0.0 when no match exists."""
    tweet_authors = ["Alice Smith"]
    paper_authors = "Bob Jones, Carl Doe"
    
    similarity = compute_author_similarity(tweet_authors, paper_authors)
    
    # No match should return low similarity
    assert similarity < 0.3


def test_compute_method_match_with_method_terms():
    """D-03: Method/Finding signal has secondary priority."""
    tweet_terms = ["regression", "neural network"]
    paper_text = "This study uses regression analysis and neural network models."
    
    match_score = compute_method_match(tweet_terms, paper_text)
    
    # Should have high match score when terms appear in paper
    assert match_score > 0.5


def test_compute_method_match_no_terms():
    """D-03: Method match returns 0.0 when no terms match."""
    tweet_terms = ["regression", "neural network"]
    paper_text = "This study uses traditional statistical methods."
    
    match_score = compute_method_match(tweet_terms, paper_text)
    
    # No terms match, should return low score
    assert match_score < 0.3


def test_compute_venue_match_with_venue_hint():
    """D-03: Venue/Year signal has lowest priority."""
    tweet_hints = ["Nature", "2023"]
    paper_venue = "Nature Communications 2023"
    
    match_score = compute_venue_match(tweet_hints, paper_venue)
    
    # Should match venue substring
    assert match_score > 0.0


def test_compute_venue_match_no_match():
    """D-03: Venue match returns 0.0 when no hints match."""
    tweet_hints = ["Nature", "2023"]
    paper_venue = "Science 2022"
    
    match_score = compute_venue_match(tweet_hints, paper_venue)
    
    # No match, should return 0.0
    assert match_score == 0.0


def test_compute_disambiguation_score_prioritizes_author():
    """D-03: Disambiguation scoring prioritizes Author > Method > Venue."""
    config = RetrievalConfig()
    evidence = TweetEvidence(
        query_text_for_embedding="query",
        candidate_authors=["Alice Smith"],
        method_terms=["regression"],
        finding_terms=[],
        time_or_venue_hints=["Nature"],
    )
    
    metadata_by_pubkey = {
        "paper1": {
            "authors": "Alice Smith",
            "abstract": "unrelated content",
            "title": "Paper",
            "venue": "unrelated",
        }
    }
    
    score = compute_disambiguation_score(
        pubkey="paper1",
        evidence=evidence,
        metadata_by_pubkey=metadata_by_pubkey,
        config=config,
    )
    
    # Should have positive score due to author match
    assert score > 0.0


def test_disambiguate_title_collisions_only_affects_collision_groups():
    """D-05: Disambiguation only reorders within collision groups."""
    config = RetrievalConfig()
    evidence = TweetEvidence(
        query_text_for_embedding="query",
        candidate_authors=["Alice Smith"],
        method_terms=["regression"],
        finding_terms=[],
        time_or_venue_hints=[],
    )
    
    metadata_by_pubkey = {
        "paper1": {
            "title": "Climate Model Study",
            "authors": "Alice Smith",
            "abstract": "regression analysis",
            "venue": "",
        },
        "paper2": {
            "title": "Climate Model Study",  # Duplicate title
            "authors": "Bob Jones",
            "abstract": "different method",
            "venue": "",
        },
        "paper3": {
            "title": "Unique Paper Title",
            "authors": "Carl Doe",
            "abstract": "unrelated",
            "venue": "",
        },
    }
    
    # Original order: paper2, paper1, paper3
    # After disambiguation: paper1 (author match) should rank higher than paper2
    # paper3 should stay at position 3 (not in collision group)
    reranked = disambiguate_title_collisions(
        candidates=["paper2", "paper1", "paper3"],
        evidence=evidence,
        metadata_by_pubkey=metadata_by_pubkey,
        config=config,
    )
    
    # paper1 should rank higher than paper2 due to author match
    assert reranked.index("paper1") < reranked.index("paper2")
    # paper3 should remain at the end (not affected by disambiguation)
    assert reranked[-1] == "paper3"


def test_disambiguate_title_collisions_preserves_semantic_order_on_tie():
    """D-05, D-06: Preserve semantic reranker order when disambiguation scores tie."""
    config = RetrievalConfig()
    evidence = TweetEvidence(
        query_text_for_embedding="query",
        candidate_authors=[],  # No author hints
        method_terms=[],  # No method hints
        finding_terms=[],
        time_or_venue_hints=[],
    )
    
    metadata_by_pubkey = {
        "paper1": {
            "title": "Climate Model Study",
            "authors": "Author A",
            "abstract": "content a",
            "venue": "",
        },
        "paper2": {
            "title": "Climate Model Study",  # Duplicate title
            "authors": "Author B",
            "abstract": "content b",
            "venue": "",
        },
    }
    
    # Original order: paper1, paper2
    # No disambiguation signals, should preserve original order (semantic reranker order)
    reranked = disambiguate_title_collisions(
        candidates=["paper1", "paper2"],
        evidence=evidence,
        metadata_by_pubkey=metadata_by_pubkey,
        config=config,
    )
    
    # Should preserve original order when scores tie
    assert reranked == ["paper1", "paper2"]


def test_disambiguate_title_collisions_no_collisions_preserves_order():
    """D-01: No disambiguation when no collisions exist."""
    config = RetrievalConfig()
    evidence = TweetEvidence(
        query_text_for_embedding="query",
        candidate_authors=["Alice Smith"],
        method_terms=[],
        finding_terms=[],
        time_or_venue_hints=[],
    )
    
    metadata_by_pubkey = {
        "paper1": {"title": "Climate Model Study", "authors": "Alice Smith", "abstract": "", "venue": ""},
        "paper2": {"title": "Weather Pattern Analysis", "authors": "Bob Jones", "abstract": "", "venue": ""},
        "paper3": {"title": "Ocean Temperature Trends", "authors": "Carl Doe", "abstract": "", "venue": ""},
    }
    
    # No duplicate titles, should preserve exact input order
    reranked = disambiguate_title_collisions(
        candidates=["paper1", "paper2", "paper3"],
        evidence=evidence,
        metadata_by_pubkey=metadata_by_pubkey,
        config=config,
    )
    
    assert reranked == ["paper1", "paper2", "paper3"]
