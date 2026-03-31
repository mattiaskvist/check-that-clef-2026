# Phase 3: Duplicate Disambiguation - Context

**Gathered:** 2026-03-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Handle papers with identical or near-identical titles by detecting duplicate-title collisions in reranked results and applying secondary signal matching (author names, method terms, venue/year) to correctly distinguish and rank them.

</domain>

<decisions>
## Implementation Decisions

### Duplicate detection strategy
- **D-01:** Trigger disambiguation only when duplicate titles appear in top-K reranked results (lazy evaluation — no overhead when not needed).
- **D-02:** Use normalized title matching to detect collisions (handles casing, punctuation variants).

### Signal priority for disambiguation
- **D-03:** When duplicates are detected, prioritize signals in this order: Author > Method/Finding > Venue/Year.
- **D-04:** This is consistent with Phase 1's title+author > method/finding ordering — extending the same signal hierarchy to the disambiguation context.

### Fallback behavior
- **D-05:** When disambiguation signals tie (both papers score equally), preserve semantic reranker ordering from Phase 2.
- **D-06:** Trust Phase 2 ranking as final arbiter — no random or arbitrary tie-breaking.

### the agent's Discretion
- Exact normalization rules for title matching (lowercasing, punctuation stripping, etc.) as long as D-01/D-02 intent holds.
- Internal data structures for tracking collision groups and disambiguation scores.
- Whether to surface disambiguation diagnostics in evaluation output (nice-to-have, not required).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope and acceptance targets
- `.planning/ROADMAP.md` §Phase 3: Duplicate Disambiguation — phase goal, dependencies, and success criteria.
- `.planning/REQUIREMENTS.md` — `RNK-03` requirement and traceability table.
- `.planning/PROJECT.md` — milestone constraints (cost-awareness, subset-first experiment loop, benchmark comparability).
- `.planning/STATE.md` — current execution position and continuity notes.

### Phase 1 and Phase 2 behavior that must remain compatible
- `.planning/phases/01-query-understanding/01-CONTEXT.md` — locked extraction/normalization/subset policy decisions.
- `.planning/phases/02-semantic-reranking/02-CONTEXT.md` — locked semantic-primary ordering and tie-break decisions.
- `.planning/phases/02-semantic-reranking/02-VERIFICATION.md` — verified behaviors that Phase 3 must not regress.
- `src/clef_retrieval/pipeline.py` — current retrieval/rerank flow and Phase 2 semantic reranking integration.
- `src/clef_retrieval/reranker.py` — semantic reranker adapters and scoring interfaces.
- `src/clef_retrieval/query_policy.py` — seeded subset and promotion/no-regression helpers carried forward.

### Existing implementation assets for disambiguation
- `src/clef_retrieval/schemas.py` — `TweetEvidence` (candidate_authors, time_or_venue_hints, method_terms) and `PaperEvidence` (authors, venue, method_terms) structures.
- `src/clef_retrieval/pipeline.py` — `weighted_signal_rerank()` uses title+author matching; extend for collision detection.
- `src/clef_retrieval/config.py` — retrieval configuration model for any new disambiguation parameters.
- `tests/test_signal_weighting.py` — current ordering/constraint expectations used as baseline comparison.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TweetEvidence.candidate_authors` and `TweetEvidence.time_or_venue_hints` provide tweet-side disambiguation signals.
- `PaperEvidence.authors`, `PaperEvidence.venue`, `PaperEvidence.method_terms` provide paper-side matching targets.
- `weighted_signal_rerank()` in `src/clef_retrieval/pipeline.py` already computes author and method term matches — can be extended for collision-aware disambiguation.
- `normalize_tokens_for_language()` provides language-aware normalization for signal matching.

### Established Patterns
- Config-first behavior control via `RetrievalConfig` with Pydantic validation.
- Deterministic, test-first policy evolution with focused unit/CLI tests.
- Phase 2 established semantic-primary ordering with weighted tie-break — disambiguation should integrate as a post-rerank refinement step.

### Integration Points
- Disambiguation logic should apply after semantic reranking, before final top-5 output.
- `rank_from_query_embedding()` in `src/clef_retrieval/pipeline.py` is the primary insertion point.
- Tests should verify disambiguation only activates when collisions exist and does not regress non-collision cases.

</code_context>

<specifics>
## Specific Ideas

- Only compute disambiguation when title collisions exist in top-K — lazy evaluation minimizes overhead.
- Extend existing signal matching rather than building parallel disambiguation system.
- Surface collision detection in diagnostics (nice-to-have) to understand how often duplicates affect rankings.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-duplicate-disambiguation*
*Context gathered: 2026-03-31*
