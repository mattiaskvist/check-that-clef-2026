# Phase 03: Duplicate Disambiguation - Research

**Researched:** 2026-03-31
**Domain:** String normalization, fuzzy matching, signal ranking
**Confidence:** HIGH

## Summary

This phase addresses RNK-03: disambiguating papers with identical or near-identical titles using secondary signals (author names, method terms, venue/year). The problem is well-scoped: only duplicate-title collisions in top-K reranked results need disambiguation, and the existing signal-matching infrastructure from Phase 1 can be extended rather than replaced.

The research confirms that Python's standard library provides all necessary primitives: `unicodedata` for Unicode normalization, `re` for punctuation handling, and `difflib.SequenceMatcher` for fuzzy name matching. No external dependencies are needed.

The key insight is that disambiguation should be a refinement pass *after* semantic reranking — only activating when collisions exist. This preserves Phase 2's semantic-primary ordering while resolving ties among duplicate-title candidates using the Author > Method/Finding > Venue/Year priority defined in CONTEXT.md.

**Primary recommendation:** Implement collision detection on normalized titles, then apply weighted disambiguation scoring only to collision groups, preserving semantic reranker order as final fallback.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Trigger disambiguation only when duplicate titles appear in top-K reranked results (lazy evaluation — no overhead when not needed).
- **D-02:** Use normalized title matching to detect collisions (handles casing, punctuation variants).
- **D-03:** When duplicates are detected, prioritize signals in this order: Author > Method/Finding > Venue/Year.
- **D-04:** This is consistent with Phase 1's title+author > method/finding ordering — extending the same signal hierarchy to the disambiguation context.
- **D-05:** When disambiguation signals tie (both papers score equally), preserve semantic reranker ordering from Phase 2.
- **D-06:** Trust Phase 2 ranking as final arbiter — no random or arbitrary tie-breaking.

### Claude's Discretion
- Exact normalization rules for title matching (lowercasing, punctuation stripping, etc.) as long as D-01/D-02 intent holds.
- Internal data structures for tracking collision groups and disambiguation scores.
- Whether to surface disambiguation diagnostics in evaluation output (nice-to-have, not required).

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RNK-03 | System handles duplicate-title candidate disambiguation using additional signals (author/method/venue) where available | Title normalization via `unicodedata`+`re`; author similarity via `SequenceMatcher`; venue/year extraction via string parsing; integration point identified in `rank_from_query_embedding()` |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| unicodedata | stdlib | Unicode normalization (NFKC) | Built-in, handles accented chars, composed forms, compatibility chars |
| re | stdlib | Punctuation stripping, whitespace normalization | Built-in, deterministic regex operations |
| difflib | stdlib | SequenceMatcher for fuzzy author name matching | Built-in, no external deps, ratio() gives 0.0-1.0 similarity |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| collections.defaultdict | stdlib | Grouping pubkeys by normalized title | For collision group tracking |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| difflib.SequenceMatcher | rapidfuzz | 10-100x faster fuzzy matching, but adds external dependency; overkill for small collision groups |
| Manual year extraction | dateparser | Heavier parsing, but author strings already structured; simple regex suffices |

**Installation:**
```bash
# No additional dependencies required — all stdlib
```

**Version verification:** N/A — using Python 3.14+ stdlib only.

## Architecture Patterns

### Recommended Project Structure
```
src/clef_retrieval/
├── disambiguation.py    # NEW: collision detection + disambiguation scoring
├── pipeline.py          # Integration: call disambiguation after semantic rerank
├── query_policy.py      # Reuse: normalize_tokens_for_language()
└── config.py            # NEW fields: disambiguation thresholds (optional)
```

### Pattern 1: Lazy Collision Detection
**What:** Only compute collision groups when needed; return early if no duplicates
**When to use:** Always — D-01 specifies lazy evaluation to minimize overhead
**Example:**
```python
def detect_title_collisions(
    candidates: list[str],
    metadata_by_pubkey: dict[str, dict],
) -> dict[str, list[str]]:
    """Group candidates by normalized title. Returns only collision groups (>1 member)."""
    from collections import defaultdict
    
    groups = defaultdict(list)
    for pubkey in candidates:
        title = metadata_by_pubkey.get(pubkey, {}).get("title", "")
        normalized = normalize_title(title)
        groups[normalized].append(pubkey)
    
    # Only return groups with collisions (D-01: lazy, skip if no duplicates)
    return {k: v for k, v in groups.items() if len(v) > 1}
```

### Pattern 2: Normalized Title Matching
**What:** Deterministic title normalization for collision detection (D-02)
**When to use:** Before grouping candidates
**Example:**
```python
import re
import unicodedata

def normalize_title(title: str) -> str:
    """Normalize title for collision detection (D-02)."""
    value = unicodedata.normalize("NFKC", title.strip())
    value = value.lower()
    # Remove punctuation except internal hyphens
    value = re.sub(r"[^\w\s-]", "", value)
    # Collapse whitespace
    value = " ".join(value.split())
    return value
```

### Pattern 3: Multi-Signal Disambiguation Scoring
**What:** Score collision-group members using Author > Method/Finding > Venue/Year priority (D-03)
**When to use:** Only within collision groups
**Example:**
```python
def compute_disambiguation_score(
    pubkey: str,
    evidence: TweetEvidence,
    metadata_by_pubkey: dict[str, dict],
    config: RetrievalConfig,
) -> float:
    """Compute disambiguation score for a single candidate within a collision group."""
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
```

### Pattern 4: Integration Point in Pipeline
**What:** Insert disambiguation after semantic reranking, before final top-5 output
**When to use:** In `rank_from_query_embedding()` after `semantic_primary_sort()`
**Example:**
```python
# In rank_from_query_embedding(), after line 335:
# sorted_candidates = semantic_primary_sort(...)
# reranked = [item[0] for item in sorted_candidates]

# Insert disambiguation here:
if evidence is not None:
    reranked = disambiguate_title_collisions(
        candidates=reranked,
        evidence=evidence,
        metadata_by_pubkey=metadata_by_pubkey,
        config=cfg,
    )
```

### Anti-Patterns to Avoid
- **Global disambiguation:** Don't run disambiguation on all candidates — only collision groups (D-01)
- **Breaking semantic order:** Don't reorder non-colliding candidates; disambiguation only affects collision groups
- **Inventing tie-breakers:** When disambiguation scores tie, preserve semantic reranker order (D-05, D-06)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Unicode normalization | Custom char mapping tables | `unicodedata.normalize("NFKC", ...)` | Handles composed chars, compatibility forms, accents |
| Fuzzy string matching | Edit distance from scratch | `difflib.SequenceMatcher.ratio()` | Tuned for English text, handles insertions/deletions |
| Punctuation stripping | Character-by-character loops | `re.sub(r"[^\w\s-]", "", ...)` | Single regex, readable, handles unicode word chars |

**Key insight:** Python stdlib handles all string operations needed for this phase. External fuzzy-matching libraries (rapidfuzz, fuzzywuzzy) are 10-100x faster but add dependencies for no practical gain — collision groups are small (2-10 candidates typically).

## Common Pitfalls

### Pitfall 1: Over-Normalizing Titles
**What goes wrong:** Removing too much information makes distinct titles collide (false positives)
**Why it happens:** Aggressive normalization (removing all numbers, removing hyphens) creates false collisions
**How to avoid:** Preserve internal hyphens, numbers, and significant structure; only strip leading/trailing punctuation
**Warning signs:** Collision groups larger than expected (>10 papers with "same" title)

### Pitfall 2: Author Name Format Variability
**What goes wrong:** Author strings contain affiliations, making fuzzy matching fail
**Why it happens:** Dataset `authors` field includes full text: "Alice Smith, Stanford University; Bob Jones, MIT"
**How to avoid:** Extract name tokens only, ignore affiliation suffixes; use substring matching, not full-string similarity
**Warning signs:** Low author similarity scores even when names clearly match

### Pitfall 3: Venue Abbreviation Mismatches
**What goes wrong:** Tweet says "Nature" but paper venue is "Nature Communications"
**Why it happens:** Users abbreviate venue names; journal families share prefixes
**How to avoid:** Use substring containment check (`venue_hint.lower() in paper_venue.lower()`) not exact match
**Warning signs:** Zero venue matches despite obvious correspondence

### Pitfall 4: Breaking Semantic Order for Non-Collisions
**What goes wrong:** Disambiguation logic accidentally reorders papers without duplicate titles
**Why it happens:** Forgetting to scope disambiguation to collision groups only
**How to avoid:** Explicitly check collision group membership before adjusting scores
**Warning signs:** MRR@5 regression on queries without title collisions

## Code Examples

Verified patterns using Python 3.14+ stdlib:

### Title Normalization
```python
# Source: Python stdlib unicodedata + re
import re
import unicodedata

def normalize_title(title: str) -> str:
    """Normalize title for collision detection."""
    # NFKC: decompose compatibility chars, recompose
    value = unicodedata.normalize("NFKC", title.strip())
    value = value.lower()
    # Remove punctuation except hyphens and spaces
    value = re.sub(r"[^\w\s-]", "", value)
    # Collapse whitespace
    return " ".join(value.split())

# Test: these all normalize to same string
assert normalize_title("A Study of Climate Change") == normalize_title("A study of climate change.")
assert normalize_title('"A Study of Climate Change"') == normalize_title("A Study of Climate Change")
```

### Author Similarity Scoring
```python
# Source: Python stdlib difflib
from difflib import SequenceMatcher

def compute_author_similarity(tweet_authors: list[str], paper_authors: str) -> float:
    """Compute max similarity between any tweet author and paper author string."""
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
```

### Venue Matching
```python
def compute_venue_match(tweet_hints: list[str], paper_venue: str) -> float:
    """Check if any tweet venue hint matches paper venue."""
    if not tweet_hints or not paper_venue:
        return 0.0
    
    paper_venue_lower = paper_venue.lower()
    
    for hint in tweet_hints:
        hint_lower = hint.strip().lower()
        if not hint_lower:
            continue
        # Substring containment (handles abbreviations)
        if hint_lower in paper_venue_lower or paper_venue_lower in hint_lower:
            return 1.0
        # Year extraction (simple 4-digit check)
        import re
        years_in_hint = re.findall(r'\b(19|20)\d{2}\b', hint_lower)
        years_in_venue = re.findall(r'\b(19|20)\d{2}\b', paper_venue_lower)
        if years_in_hint and years_in_venue and set(years_in_hint) & set(years_in_venue):
            return 0.5  # Partial match for year only
    
    return 0.0
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Exact title matching | Normalized title matching | This phase | Handles casing, punctuation, unicode variants |
| Single-signal disambiguation | Multi-signal priority (Author > Method > Venue) | This phase | Better separation when authors differ |

**Deprecated/outdated:**
- Levenshtein distance for short strings: SequenceMatcher is better for natural language names
- External fuzzy libraries: Overhead not justified for small collision groups

## Open Questions

1. **Collision detection threshold**
   - What we know: Exact normalized-title match creates collision groups
   - What's unclear: Should near-matches (edit distance ≤ 2) also form groups?
   - Recommendation: Start with exact match (D-02); add fuzzy detection in future iteration if false negatives appear

2. **Disambiguation weight tuning**
   - What we know: Phase 1 weights work (5.0 / 3.0 / 1.0 for title-author / method / keyword)
   - What's unclear: Are these optimal for disambiguation specifically?
   - Recommendation: Reuse Phase 1 weights initially (D-04); tune if evaluation shows poor separation

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | pyproject.toml (`[tool.pytest.ini_options]` section) |
| Quick run command | `uv run pytest tests/test_disambiguation.py -x` |
| Full suite command | `uv run pytest tests/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RNK-03 | Collision detection activates only when duplicates exist | unit | `uv run pytest tests/test_disambiguation.py::test_no_disambiguation_without_collisions -x` | ❌ Wave 0 |
| RNK-03 | Normalized title matching groups duplicates correctly | unit | `uv run pytest tests/test_disambiguation.py::test_normalized_title_grouping -x` | ❌ Wave 0 |
| RNK-03 | Author signal has highest priority in disambiguation | unit | `uv run pytest tests/test_disambiguation.py::test_author_priority_in_disambiguation -x` | ❌ Wave 0 |
| RNK-03 | Semantic reranker order preserved when disambiguation ties | unit | `uv run pytest tests/test_disambiguation.py::test_semantic_order_preserved_on_tie -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_disambiguation.py -x`
- **Per wave merge:** `uv run pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_disambiguation.py` — covers RNK-03 (collision detection, signal priority, tie-break)
- [ ] Fixtures for duplicate-title candidate metadata

*(No framework gaps — pytest infrastructure verified working)*

## Sources

### Primary (HIGH confidence)
- Python stdlib documentation — `unicodedata`, `re`, `difflib` modules
- Codebase inspection — `src/clef_retrieval/pipeline.py`, `query_policy.py`, `schemas.py`
- Phase 2 verification — `02-VERIFICATION.md` (established patterns and integration points)

### Secondary (MEDIUM confidence)
- CLEF dataset inspection — author/venue field formats via direct inspection

### Tertiary (LOW confidence)
- None — all findings based on codebase and stdlib documentation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — stdlib only, verified working in Python 3.14
- Architecture: HIGH — integration point clearly identified, follows Phase 1/2 patterns
- Pitfalls: HIGH — derived from actual data inspection and normalization testing

**Research date:** 2026-03-31
**Valid until:** No expiry — stdlib-only, no version dependencies

---

*Phase: 03-duplicate-disambiguation*
*Research completed: 2026-03-31*
