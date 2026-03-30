---
phase: 01-query-understanding
plan: 01
subsystem: retrieval
tags: [query-policy, config, deterministic-subsets, testing]
requires: []
provides:
  - Strict extraction gate policy configuration and helper contracts
  - Language-aware token normalization for en/de/fr policy path
  - Seeded deterministic subset selection and promotion decision rules
affects: [main.py, src/clef_retrieval/pipeline.py, phase-01-followup-plans]
tech-stack:
  added: []
  patterns: [policy-module helpers, config-driven query-understanding thresholds]
key-files:
  created:
    - src/clef_retrieval/query_policy.py
    - tests/test_query_policy.py
    - tests/test_subset_policy.py
  modified:
    - src/clef_retrieval/config.py
    - tests/test_config.py
key-decisions:
  - "Enforced D-03 ordering with config validation: title+author > method/finding > keywords."
  - "Made subset policy deterministic through explicit seed and sorted sampled indices."
patterns-established:
  - "Pure policy helpers: no network/model side effects in query policy functions."
  - "Locked decision semantics captured directly in focused unit tests."
requirements-completed: [QRY-01, QRY-04, OPS-01]
duration: 5min
completed: 2026-03-30
---

# Phase 1 Plan 1: Query Policy Contracts Summary

**Config-driven strict extraction gating, multilingual normalization, and deterministic subset/promotion policy helpers for Phase 1 query understanding.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-30T23:30:00Z
- **Completed:** 2026-03-30T23:34:53Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- Added retrieval config policy fields for strict gate thresholds, weighting order, normalization mode, and subset controls.
- Introduced pure `query_policy.py` helpers for extraction gate, token normalization, deterministic subsets, and promotion decisions.
- Added focused policy tests covering D-02 through D-07 behavior expectations.

## Task Commits

1. **Task 1: Add strict policy config and ordering constants** - `0a54678` (feat)
2. **Task 2: Create query policy module for gate, normalization, subset, and promotion contracts** - `adb5591` (feat)
3. **Task 3: Add policy-focused tests for strict gate and seeded subset reproducibility** - `cd4a4a1` (test)

## Files Created/Modified
- `src/clef_retrieval/config.py` - Added policy knobs and ordering validator.
- `src/clef_retrieval/query_policy.py` - New deterministic query policy helper module.
- `tests/test_config.py` - Added assertions for new defaults and weight-order validation.
- `tests/test_query_policy.py` - Added strict gate, normalization, and promotion tests.
- `tests/test_subset_policy.py` - Added deterministic seeded subset tests.

## Decisions Made
- Kept gate acceptance strict by requiring both title and author thresholds, plus minimum filled key fields.
- Implemented normalization as deterministic text normalization primitives per language policy needs.
- Promotion helper blocks advancement on any per-language regression even when macro score improves.

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Query policy contracts and tests are in place for downstream pipeline/main integration plans.
- No blockers identified for Phase 1 Plan 2.

## Self-Check: PASSED
- Confirmed summary file exists.
- Confirmed task commits exist in git history: `0a54678`, `adb5591`, `cd4a4a1`.
