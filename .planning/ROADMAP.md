# Roadmap: CLEF 2026 Task 1 Retrieval Improvement

## Overview

This roadmap delivers measurable MRR@5 improvements over the current baseline through three focused phases: strengthening query understanding to extract better retrieval signals from noisy tweets, replacing naive lexical reranking with semantic cross-encoders for multilingual precision, and handling duplicate-title edge cases through multi-signal disambiguation. Each phase is independently verifiable on the dev split and builds incrementally toward the core goal.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Query Understanding** - Extract and utilize rich signals from tweets for retrieval (completed 2026-03-30)
- [x] **Phase 2: Semantic Reranking** - Replace lexical reranking with multilingual cross-encoder (completed 2026-03-31)
- [ ] **Phase 3: Duplicate Disambiguation** - Handle duplicate-title papers with multi-signal matching

## Phase Details

### Phase 1: Query Understanding
**Goal**: System extracts and utilizes rich structured signals from tweets to improve retrieval recall
**Depends on**: Nothing (first phase)
**Requirements**: QRY-01, QRY-02, QRY-03, QRY-04, QRY-05, EVAL-01, EVAL-02, OPS-01, OPS-02, OPS-03
**Success Criteria** (what must be TRUE):
  1. System extracts title fragments, candidate authors, method terms, and finding terms from tweets with measurable parse success rate
  2. System reports per-language (en/de/fr) baseline and experiment MRR@5 metrics for reproducible comparison
  3. System uses extracted title fragments and author names as ranking signals beyond raw embedding similarity
  4. System enforces negative constraints from tweets to filter obvious false-positive candidates
  5. System processes predictions and evaluations with cached reuse to avoid redundant LLM/embedding calls
  6. System supports subset-first experiment loops (limited queries) before full-split runs to reduce iteration latency
**Plans**: 3 plans
Plans:
- [ ] 01-01-PLAN.md — Define strict gate, language normalization, and seeded subset policy contracts
- [ ] 01-02-PLAN.md — Implement gate-aware weighted reranking with negative-constraint enforcement
- [ ] 01-03-PLAN.md — Add CLI telemetry, multilingual metrics, subset promotion gate, and cost guardrails

### Phase 2: Semantic Reranking
**Goal**: System reranks top-K candidates using semantic similarity rather than lexical overlap
**Depends on**: Phase 1
**Requirements**: RNK-01, RNK-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. System integrates multilingual cross-encoder (Jina Reranker v2 or BGE v2-m3) to rerank dense retrieval candidates
  2. System preserves or improves MRR@5 across all three languages (en/de/fr) compared to lexical baseline
  3. System provides stage-level diagnostics showing retrieval recall-at-K and reranking uplift to localize bottlenecks
**Plans**: 2 plans
Plans:
- [x] 02-01-PLAN.md — Implement backend-swappable semantic reranking integration with semantic-primary ordering and weighted tie-break.
- [x] 02-02-PLAN.md — Add stage diagnostics (Recall@K, uplift overall/per-language, latency/throughput) in evaluate output.

### Phase 3: Duplicate Disambiguation
**Goal**: System correctly ranks papers with duplicate titles using multi-signal matching
**Depends on**: Phase 2
**Requirements**: RNK-03
**Success Criteria** (what must be TRUE):
  1. System detects duplicate-title candidates in retrieval results
  2. System uses author names, method terms, and venue/year information to disambiguate duplicate titles
  3. System reduces duplicate-title ranking errors measurably on dev set queries affected by title collisions
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Query Understanding | 3/3 | Complete   | 2026-03-30 |
| 2. Semantic Reranking | 2/2 | Complete   | 2026-03-31 |
| 3. Duplicate Disambiguation | 0/TBD | Not started | - |
