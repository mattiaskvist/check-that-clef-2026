---
phase: 02-semantic-reranking
plan: 02
subsystem: retrieval
tags: [diagnostics, recall, uplift, latency, evaluate]

# Dependency graph
requires:
  - phase: 02-semantic-reranking
    plan: 01
    provides: semantic reranker protocol and pipeline integration
provides:
  - Stage-level diagnostics in evaluate command (Recall@K, uplift, latency)
  - Per-language uplift delta reporting
  - Bottleneck localization (retrieval vs reranker)
affects: [evaluation, experiment-loop, main-cli]

# Tech tracking
tech-stack:
  added: []
  patterns: [Stage diagnostics header with mode label, bottleneck localization via Recall@K vs MRR comparison]

key-files:
  created:
    - tests/test_stage_diagnostics.py
  modified:
    - main.py
    - tests/test_language_metrics_reporting.py

key-decisions:
  - "Recall@5 as primary retrieval-stage diagnostic metric"
  - "Mode-aware latency labels (eval_latency vs rerank_latency)"
  - "Uplift reporting gated on promotion baseline flags for opt-in clarity"

patterns-established:
  - "_print_diagnostics_header: consistent diagnostics section with rerank_top_k and mode"
  - "_compute_recall_at_k: dataset-aware recall computation with HF dataset loading"
  - "_format_uplift: signed delta format for uplift reporting"

requirements-completed: [EVAL-03, RNK-02]

# Metrics
duration: 5min
completed: 2026-03-31
---

# Phase 02 Plan 02: Evaluate Stage Diagnostics Summary

**Stage-level diagnostics for semantic reranking with Recall@K, MRR uplift, and latency reporting**

## Performance

- **Duration:** 4 min 47s
- **Started:** 2026-03-31T08:49:20Z
- **Completed:** 2026-03-31T08:54:07Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Implemented Recall@K diagnostics for dense retrieval stage (D-07)
- Added semantic MRR@5 uplift delta reporting overall and per-language (D-08)
- Added reranker latency/throughput diagnostics with mode-aware labels (D-09)
- Enhanced diagnostics header with rerank_top_k and cached/recompute mode
- Phase 1 promotion gate semantics preserved and tested

## Task Commits

Each task was committed atomically:

1. **Task 1: Add failing diagnostics tests (RED)** - `9cd9f4e` (test)
2. **Task 2: Implement evaluate-stage diagnostics (GREEN)** - `1804531` (feat)
3. **Task 3: Harden diagnostics defaults and ergonomics** - `e4ca611` (refactor)

## Files Created/Modified
- `main.py` - Added _compute_recall_at_k, _format_uplift, _print_diagnostics_header, extended _evaluate
- `tests/test_stage_diagnostics.py` - New diagnostics contract tests (D-07, D-08, D-09)
- `tests/test_language_metrics_reporting.py` - Phase 1 compatibility tests for promotion semantics

## Decisions Made
- Recall@5 chosen as primary diagnostic since it directly shows if correct answers reach final ranking
- Mode-aware latency label distinguishes eval-only timing from full rerank timing
- Uplift reporting requires all baseline flags to avoid partial/confusing output

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required. Diagnostics use existing prediction cache and HuggingFace datasets.

## Next Phase Readiness
- Semantic reranking Phase 2 complete with full diagnostics
- Evaluate command now clearly separates retrieval bottlenecks from reranker bottlenecks
- Ready for Phase 3 or full experiment runs with visible quality/cost tradeoffs

---
*Phase: 02-semantic-reranking*
*Completed: 2026-03-31*

## Self-Check: PASSED

All files exist and all commits verified.
