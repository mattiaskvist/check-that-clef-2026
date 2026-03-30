---
phase: 01-query-understanding
plan: 03
subsystem: testing
tags: [cli, telemetry, multilingual-metrics, subset-policy, cost-guardrails]
requires:
  - phase: 01-query-understanding
    provides: "Strict policy helpers and weighted extraction outcomes from Plans 01-01 and 01-02"
provides:
  - CLI tests for multilingual metrics reporting, seeded subset reproducibility, promotion gate, and cost transparency
  - Predict/evaluate controls for seeded subset runs with deterministic selection
  - Extraction outcome telemetry in CLI output with accepted/rejected/fallback rates
  - Query API usage estimate output and cost guardrail enforcement with override mode
affects: [main.py, tests/test_cli.py, phase-01-verification]
tech-stack:
  added: []
  patterns: [telemetry-first cli output, deterministic subset sampling, no-regression promotion gate]
key-files:
  created:
    - tests/test_language_metrics_reporting.py
    - tests/test_cost_transparency.py
  modified:
    - main.py
    - tests/test_cli.py
    - tests/test_main_progress.py
key-decisions:
  - "Kept cached evaluation behavior intact and layered multilingual metrics/promotion checks as optional flags."
  - "Counted skip-query-extraction rows as fallback telemetry to preserve deterministic extraction outcome totals."
patterns-established:
  - "Predict now prints usage estimates and guardrail status before expensive model calls."
  - "Subset sampling is deterministic from seed + sorted sampled indices for reproducible experiment loops."
requirements-completed: [QRY-02, QRY-04, EVAL-01, EVAL-02, OPS-01, OPS-02, OPS-03]
duration: 41min
completed: 2026-03-31
---

# Phase 1 Plan 3: CLI Telemetry, Promotion Gate, and Cost Guardrails Summary

**CLI now supports deterministic subset experimentation, multilingual promotion gating, extraction telemetry, and explicit API/cost transparency for Phase 1 runs.**

## Performance

- **Duration:** 41 min
- **Started:** 2026-03-31T00:21:00Z
- **Completed:** 2026-03-31T01:02:00Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- Added test coverage for multilingual metrics output, seeded subset reproducibility, promotion no-regression behavior, and cost guardrails.
- Implemented seeded subset controls, extraction outcome telemetry, per-language reporting, and promotion gate logic in CLI flows.
- Added model/API usage estimation, projected cost output, and configurable guardrail enforcement with override warning mode.

## Task Commits

1. **Task 1: Add CLI tests for multilingual metrics, subset-seed reproducibility, promotion gate, and cost transparency** - `70a20e4` (test)
2. **Task 2: Implement CLI seeded subset loop, extraction telemetry, and promotion/no-regression gate** - `d055b23` (feat)
3. **Task 3: Implement model/API usage transparency and practical cost guardrails** - `fa0b770` (feat)

## Files Created/Modified
- `main.py` - Added subset flags, extraction telemetry, multilingual metrics/promotion logic, usage estimate, and cost guardrail behavior.
- `tests/test_cli.py` - Added CLI flag acceptance coverage for new subset and multilingual metric switches.
- `tests/test_language_metrics_reporting.py` - Added deterministic subset, per-language metrics, and promotion block tests.
- `tests/test_cost_transparency.py` - Added usage estimate and guardrail behavior tests.
- `tests/test_main_progress.py` - Updated compatibility monkeypatch to align with new extraction outcome helper.

## Decisions Made
- Preserved existing cached-evaluation path semantics (`--recompute` still controls fresh prediction generation).
- Implemented multilingual promotion gate as an explicit CLI decision output (`PROMOTE`/`BLOCKED`) tied to baseline inputs.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed progress test monkeypatch after query helper refactor**
- **Found during:** Task 3 verification (`uv run pytest -q`)
- **Issue:** `tests/test_main_progress.py` patched removed `build_query_embedding_text` symbol and failed full-suite verification.
- **Fix:** Updated test to monkeypatch `_build_query_text_with_outcome`, matching current helper contract.
- **Files modified:** `tests/test_main_progress.py`
- **Verification:** `uv run pytest -q` passed (68 passed)
- **Committed in:** `fa0b770`

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** No scope creep; fix was required to keep regression suite aligned with implemented CLI behavior.

## Authentication Gates
None.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 1 CLI behavior is now reproducible and observable for subset-first experimentation.
- Cost and promotion checks are codified and test-backed for verifier review.

## Known Stubs
None detected in modified files.

## Self-Check: PASSED
- Confirmed summary file exists: `.planning/phases/01-query-understanding/01-query-understanding-03-SUMMARY.md`
- Confirmed task commits exist in git history: `70a20e4`, `d055b23`, `fa0b770`
