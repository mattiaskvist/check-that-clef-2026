---
phase: 01-query-understanding
plan: 02
subsystem: retrieval
tags: [pipeline, query-understanding, reranking, telemetry]
requires:
  - phase: 01-query-understanding
    provides: "Strict gate/normalization/subset policy helpers from Plan 01"
provides:
  - Gate-aware query text construction with strict extraction fallback
  - Deterministic weighted reranking with ordered title-author > method-finding > keywords signals
  - Negative-constraint penalty handling and extraction outcome metadata helper
affects: [main.py, src/clef_retrieval/pipeline.py, phase-01-plan-03]
tech-stack:
  added: []
  patterns: [policy-driven fallback classification, deterministic weighted scoring]
key-files:
  created:
    - tests/test_signal_weighting.py
  modified:
    - src/clef_retrieval/pipeline.py
    - tests/test_pipeline.py
key-decisions:
  - "Kept build_query_embedding_text backward compatible and introduced companion metadata helper for telemetry."
  - "Used candidate-position tie-breaks so weighted rerank remains deterministic for equal scores."
patterns-established:
  - "Extraction outcomes are explicitly labeled as parsed+accepted, parsed+rejected, or error->fallback."
  - "Weighted signal reranking relies on normalized evidence/doc terms and config-controlled priorities."
requirements-completed: [QRY-01, QRY-02, QRY-03, QRY-05]
duration: 18min
completed: 2026-03-31
---

# Phase 1 Plan 2: Gate-Aware Query Construction and Weighted Reranking Summary

**Pipeline now uses strict extraction-gate fallback plus deterministic weighted evidence reranking with negative-constraint penalties and extraction-outcome labels.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-03-31T00:02:00Z
- **Completed:** 2026-03-31T00:20:00Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Added failing-first tests for strict gate rejection, weighted signal ordering, and negative constraints.
- Implemented policy-aware extraction outcome handling and deterministic weighted reranking in `pipeline.py`.
- Added compatibility-preserving metadata helper contract tests for telemetry-ready extraction outcomes.

## Task Commits

1. **Task 1: Add failing tests for gated extraction, signal weighting order, and negative constraints** - `d2dce50` (test)
2. **Task 2: Implement strict gate-aware query text construction and weighted reranking** - `eac437c` (feat)
3. **Task 3: Preserve backward-compatible pipeline interfaces and add extraction outcome metadata hook** - `4feba39` (test)

## Files Created/Modified
- `src/clef_retrieval/pipeline.py` - Added weighted signal reranker, gate-aware metadata helper, and compatibility-preserving query text path.
- `tests/test_pipeline.py` - Added weak extraction fallback and extraction outcome-label contract checks.
- `tests/test_signal_weighting.py` - Added explicit signal-priority and negative-constraint ranking tests.

## Decisions Made
- Preserved `build_query_embedding_text()` signature for legacy callers and routed it through the new metadata helper.
- Encoded deterministic scoring tie-break using original candidate order for stable rerank output.

## Deviations from Plan
None - plan executed exactly as written.

## Authentication Gates
None.

## Known Stubs
None detected in modified files.

## Issues Encountered
- Existing baseline test fixtures needed stronger extracted evidence fields to satisfy the strict gate; fixtures were updated to reflect accepted extraction cases.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Pipeline now exposes extraction outcome labels needed for upcoming CLI instrumentation work.
- Weighted query understanding behavior is deterministic and regression-tested for ordering and constraints.

## Self-Check: PASSED
- Confirmed summary file exists: `.planning/phases/01-query-understanding/01-query-understanding-02-SUMMARY.md`
- Confirmed task commits exist in git history: `d2dce50`, `eac437c`, `4feba39`
