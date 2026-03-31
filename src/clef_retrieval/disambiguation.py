"""Duplicate-title disambiguation logic for RNK-03."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

from .config import RetrievalConfig
from .schemas import TweetEvidence


def normalize_title(title: str) -> str:
    """
    Normalize title for collision detection (D-02).
    
    Handles casing, punctuation, and Unicode variants to detect duplicate titles.
    """
    # NFKC: decompose compatibility chars, recompose
    value = unicodedata.normalize("NFKC", title.strip())
    value = value.lower()
    # Remove punctuation except hyphens and spaces
    value = re.sub(r"[^\w\s-]", "", value)
    # Collapse whitespace
    return " ".join(value.split())


def detect_title_collisions(
    candidates: list[str],
    metadata_by_pubkey: dict[str, dict],
) -> dict[str, list[str]]:
    """
    Group candidates by normalized title. Returns only collision groups (>1 member).
    
    D-01: Lazy evaluation — only activates when duplicate titles exist.
    """
    groups = defaultdict(list)
    for pubkey in candidates:
        title = metadata_by_pubkey.get(pubkey, {}).get("title", "")
        normalized = normalize_title(title)
        groups[normalized].append(pubkey)
    
    # Only return groups with collisions (D-01: lazy, skip if no duplicates)
    return {k: v for k, v in groups.items() if len(v) > 1}


def compute_author_similarity(tweet_authors: list[str], paper_authors: str) -> float:
    """
    Compute max similarity between any tweet author and paper author string.
    
    D-03: Author signal has highest disambiguation weight.
    Uses substring containment (fast path) and fuzzy matching fallback.
    """
    if not tweet_authors or not paper_authors:
        return 0.0
    
    paper_lower = paper_authors.lower()
    max_sim = 0.0
    
    for author in tweet_authors:
        author_lower = author.strip().lower()
        # Check substring containment first (fast path)
        if author_lower in paper_lower:
            return 1.0
        # Fall back to fuzzy matching
        sim = SequenceMatcher(None, author_lower, paper_lower).ratio()
        # Partial match: check if author name appears as subsequence
        if len(author_lower) > 3:
            for word in paper_lower.split():
                word_sim = SequenceMatcher(None, author_lower, word).ratio()
                sim = max(sim, word_sim)
        max_sim = max(max_sim, sim)
    
    return max_sim


def compute_method_match(tweet_terms: list[str], paper_text: str) -> float:
    """
    Check if any tweet method/finding terms match paper text.
    
    D-03: Method/Finding signal has secondary priority.
    """
    if not tweet_terms or not paper_text:
        return 0.0
    
    paper_lower = paper_text.lower()
    matched = sum(1 for term in tweet_terms if term.lower() in paper_lower)
    
    # Return proportion of matched terms
    return matched / len(tweet_terms) if tweet_terms else 0.0


def compute_venue_match(tweet_hints: list[str], paper_venue: str) -> float:
    """
    Check if any tweet venue hint matches paper venue.
    
    D-03: Venue/Year signal has lowest priority.
    Uses substring containment (handles abbreviations) and year extraction.
    """
    if not tweet_hints or not paper_venue:
        return 0.0
    
    paper_venue_lower = paper_venue.lower()
    
    for hint in tweet_hints:
        hint_lower = hint.strip().lower()
        if not hint_lower:
            continue
        # Check if this hint is a year (4-digit number starting with 19 or 20)
        is_year = bool(re.match(r'^(19|20)\d{2}$', hint_lower))
        
        if is_year:
            # For year hints, only match if the exact year appears in venue
            if hint_lower in paper_venue_lower:
                return 0.5  # Partial match for year only
        else:
            # For non-year hints (venue names), use substring containment
            if hint_lower in paper_venue_lower or paper_venue_lower in hint_lower:
                return 1.0
    
    return 0.0


def compute_disambiguation_score(
    pubkey: str,
    evidence: TweetEvidence,
    metadata_by_pubkey: dict[str, dict],
    config: RetrievalConfig,
) -> float:
    """
    Compute disambiguation score for a single candidate within a collision group.
    
    D-03: Prioritizes signals in this order: Author > Method/Finding > Venue/Year.
    Uses config weights consistent with Phase 1 signal hierarchy.
    """
    meta = metadata_by_pubkey.get(pubkey, {})
    
    # Signal 1: Author match (highest weight per D-03)
    author_score = compute_author_similarity(
        tweet_authors=evidence.candidate_authors,
        paper_authors=meta.get("authors", ""),
    )
    
    # Signal 2: Method/Finding match (second priority)
    method_score = compute_method_match(
        tweet_terms=evidence.method_terms + evidence.finding_terms,
        paper_text=f"{meta.get('abstract', '')} {meta.get('title', '')}",
    )
    
    # Signal 3: Venue/Year match (lowest priority)
    venue_score = compute_venue_match(
        tweet_hints=evidence.time_or_venue_hints,
        paper_venue=meta.get("venue", ""),
    )
    
    # Weighted combination (same hierarchy as Phase 1 weighted scoring)
    return (
        config.weight_title_author * author_score +
        config.weight_method_finding * method_score +
        config.weight_keywords * venue_score  # Reuse lowest weight for venue
    )


def disambiguate_title_collisions(
    candidates: list[str],
    evidence: TweetEvidence,
    metadata_by_pubkey: dict[str, dict],
    config: RetrievalConfig,
) -> list[str]:
    """
    Apply disambiguation scoring only to collision groups.
    
    D-01: Only activates when duplicate titles exist (lazy evaluation).
    D-05, D-06: Preserves semantic reranker order when disambiguation scores tie.
    
    Returns reranked candidates with collision groups reordered by disambiguation signals.
    """
    # Detect collision groups
    collision_groups = detect_title_collisions(candidates, metadata_by_pubkey)
    
    # If no collisions, return candidates unchanged (D-01: lazy evaluation)
    if not collision_groups:
        return candidates
    
    # Build set of all pubkeys in collision groups for fast lookup
    collision_pubkeys = set()
    for group in collision_groups.values():
        collision_pubkeys.update(group)
    
    # Compute disambiguation scores only for collision members
    scores_by_pubkey = {}
    for pubkey in collision_pubkeys:
        scores_by_pubkey[pubkey] = compute_disambiguation_score(
            pubkey, evidence, metadata_by_pubkey, config
        )
    
    # Reorder candidates: within each collision group, sort by disambiguation score
    # Preserve original order for non-collision candidates and as tie-breaker
    reranked = []
    processed = set()
    
    for idx, pubkey in enumerate(candidates):
        if pubkey in processed:
            continue
        
        # If not in a collision group, append as-is (D-05: preserve order for non-collisions)
        if pubkey not in collision_pubkeys:
            reranked.append(pubkey)
            processed.add(pubkey)
            continue
        
        # Find the collision group this pubkey belongs to
        group_key = normalize_title(metadata_by_pubkey.get(pubkey, {}).get("title", ""))
        group_members = collision_groups.get(group_key, [])
        
        # Sort group members by disambiguation score (descending), with original order as tie-breaker
        # D-05, D-06: preserve semantic reranker order when scores tie
        group_with_scores = [
            (member, scores_by_pubkey[member], candidates.index(member))
            for member in group_members
        ]
        group_with_scores.sort(key=lambda x: (-x[1], x[2]))  # Sort by score desc, then original index
        
        # Append sorted group members
        for member, _, _ in group_with_scores:
            if member not in processed:
                reranked.append(member)
                processed.add(member)
    
    return reranked
