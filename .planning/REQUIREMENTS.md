# Requirements: CLEF 2026 Task 1 Retrieval Improvement

**Defined:** 2026-03-31
**Core Value:** Improve MRR@5 on the dev split over the repository's current baseline for Task 1 source retrieval.

## v1 Requirements

Requirements for this milestone, mapped to roadmap phases.

### Query Understanding

- [ ] **QRY-01**: System extracts structured tweet evidence (claim summary, title mentions, candidate authors, method terms, finding terms) with measurable parse-success rate.
- [ ] **QRY-02**: System logs and reports query-extraction success/fallback rates so extraction quality regressions are detectable.
- [ ] **QRY-03**: System uses extracted title fragments and candidate authors in retrieval/ranking signals rather than embedding raw tweet text alone.
- [ ] **QRY-04**: System supports language-aware query handling for en/de/fr and reports per-language dev MRR@5.
- [ ] **QRY-05**: System enforces negative constraints from tweets (when present) to reduce obvious false-positive candidates.

### Ranking Quality

- [ ] **RNK-01**: System replaces or augments lexical-only reranking with semantic reranking over top-K dense candidates.
- [ ] **RNK-02**: System reranking strategy preserves or improves multilingual behavior across en/de/fr.
- [ ] **RNK-03**: System handles duplicate-title candidate disambiguation using additional signals (author/method/venue) where available.

### Evaluation and Experimentation

- [ ] **EVAL-01**: System can evaluate from cached prediction files without recomputing embeddings/LLM calls unless explicitly requested.
- [ ] **EVAL-02**: System records baseline and experiment metrics (overall + per-language) for reproducible comparison.
- [ ] **EVAL-03**: System provides stage-level diagnostics (retrieval recall-at-K and reranking uplift indicators) to localize bottlenecks.

### Operational Efficiency

- [ ] **OPS-01**: System processes prediction/evaluation with batch controls suitable for iterative experiments.
- [ ] **OPS-02**: System keeps query-time model/API usage transparent in CLI output.
- [ ] **OPS-03**: System stays within practical cost bounds for iterative dev experimentation unless explicitly overridden.

## v2 Requirements

Deferred to future milestones.

### Advanced Retrieval

- **ADV-01**: System supports hybrid sparse+dense fusion for potential recall gains on term-heavy claims.
- **ADV-02**: System evaluates late-interaction retrieval architectures if v1 improvements plateau.
- **ADV-03**: System investigates fine-tuned/domain-adapted embeddings if baseline and v1 deltas saturate.

### Productization

- **PRD-01**: Build a web UI for experiment management and result inspection.
- **PRD-02**: Add production-grade deployment and service orchestration.

## Out of Scope

Explicitly excluded for this milestone.

| Feature | Reason |
|---------|--------|
| CLEF Task 2 or other challenge tasks | Milestone is strictly Task 1 source retrieval |
| New external datasets | Keep benchmark comparisons consistent and reproducible |
| Large infrastructure/deployment work | Not needed for current retrieval quality iteration loop |
| Product frontend/UI | Not required to improve MRR@5 on dev |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| QRY-01 | Phase 1 | Pending |
| QRY-02 | Phase 1 | Pending |
| QRY-03 | Phase 1 | Pending |
| QRY-04 | Phase 1 | Pending |
| QRY-05 | Phase 1 | Pending |
| RNK-01 | Phase 2 | Pending |
| RNK-02 | Phase 2 | Pending |
| RNK-03 | Phase 3 | Pending |
| EVAL-01 | Phase 1 | Pending |
| EVAL-02 | Phase 1 | Pending |
| EVAL-03 | Phase 2 | Pending |
| OPS-01 | Phase 1 | Pending |
| OPS-02 | Phase 1 | Pending |
| OPS-03 | Phase 1 | Pending |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-31*
*Last updated: 2026-03-31 after initial definition*
