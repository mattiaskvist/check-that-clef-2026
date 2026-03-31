---
phase: 01-query-understanding
verified: 2026-03-31T00:00:32Z
status: passed
score: 6/6 must-haves verified
---

# Phase 1: Query Understanding Verification Report

**Phase Goal:** System extracts and utilizes rich structured signals from tweets to improve retrieval recall  
**Verified:** 2026-03-31T00:00:32Z  
**Status:** passed  
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | System extracts title fragments, candidate authors, method terms, and finding terms from tweets with measurable parse success rate | ✓ VERIFIED | `pipeline.build_query_embedding_text_with_metadata()` classifies extraction as `parsed+accepted` / `parsed+rejected` / `error->fallback`; `main._predict()` prints extraction outcome counts/rate; covered in `tests/test_pipeline.py` and `tests/test_cost_transparency.py`. |
| 2 | System reports per-language (en/de/fr) baseline and experiment MRR@5 metrics for reproducible comparison | ✓ VERIFIED | `main._evaluate()` computes per-language scores and prints `MRR@5 [en/de/fr]` + overall line; validated by `tests/test_language_metrics_reporting.py`. |
| 3 | System uses extracted title fragments and author names as ranking signals beyond raw embedding similarity | ✓ VERIFIED | `pipeline.weighted_signal_rerank()` uses normalized title+author terms with highest weight (`weight_title_author`), ahead of method/finding and keywords; validated in `tests/test_signal_weighting.py`. |
| 4 | System enforces negative constraints from tweets to filter obvious false-positive candidates | ✓ VERIFIED | `pipeline.weighted_signal_rerank()` applies negative-term penalty (`-((weight_title_author + weight_method_finding) * negative_matches)`); verified by `test_weighted_signal_ranking_applies_negative_constraints_penalty`. |
| 5 | System processes predictions and evaluations with cached reuse to avoid redundant LLM/embedding calls | ✓ VERIFIED | `main._evaluate()` reuses existing prediction JSONL when present and `--recompute` is absent; validated by `test_evaluate_uses_cached_predictions_without_recomputing`. |
| 6 | System supports subset-first experiment loops (limited queries) before full-split runs to reduce iteration latency | ✓ VERIFIED | `main._select_subset_rows()` + `query_policy.select_seeded_subset_indices()` provide deterministic seeded subset selection; subset flags in CLI parser; tested in `tests/test_language_metrics_reporting.py` and `tests/test_subset_policy.py`. |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/clef_retrieval/config.py` | Policy knobs and strict ordering for query-understanding signals | ✓ VERIFIED | Exists; substantive fields for gate thresholds, weight ordering validator, subset controls. |
| `src/clef_retrieval/query_policy.py` | Strict gate, language normalization, deterministic subset, promotion helper | ✓ VERIFIED | Exists; deterministic pure helpers, no network side effects. |
| `src/clef_retrieval/pipeline.py` | Gate-aware query construction and weighted reranking with negative constraints | ✓ VERIFIED | Exists; wired to `query_policy` and `schemas`, weighted rerank + extraction metadata helper implemented. |
| `main.py` | CLI telemetry, multilingual metrics, cached evaluation, subset controls, cost guardrails | ✓ VERIFIED | Exists; parser flags and command logic implemented and tested. |
| `tests/test_pipeline.py` | Regression coverage for fallback and extraction outcomes | ✓ VERIFIED | Exists; asserts accepted/rejected/fallback behavior and query text fallback safety. |
| `tests/test_signal_weighting.py` | Signal-priority + negative-constraint ranking checks | ✓ VERIFIED | Exists; verifies ranking order and penalization behavior. |
| `tests/test_cli.py` | CLI surface and cached-eval regression checks | ✓ VERIFIED | Exists; verifies flags, errors, and cached path behavior. |
| `tests/test_language_metrics_reporting.py` | Per-language metrics and promotion gate behavior | ✓ VERIFIED | Exists; validates en/de/fr reporting and blocked promotion on regression. |
| `tests/test_cost_transparency.py` | Usage estimate and guardrail reporting checks | ✓ VERIFIED | Exists; validates projected calls/cost output and guardrail status behavior. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `main.py` | `src/clef_retrieval/query_policy.py` | `from clef_retrieval.query_policy import ...` and helper usage | ✓ WIRED | Import present (line 32); `select_seeded_subset_indices` used in `_select_subset_rows`; `should_promote_from_subset` used in promotion gate. |
| `src/clef_retrieval/pipeline.py` | `src/clef_retrieval/query_policy.py` | strict gate + normalization helper calls | ✓ WIRED | Relative import present (line 13); `extraction_gate_passes` and `normalize_tokens_for_language` invoked in query construction/reranking path. |
| `main.py::_predict_rows` | `src/clef_retrieval/pipeline.py` | extraction outcome metadata | ✓ WIRED | Imports `build_query_embedding_text_with_metadata`; `_build_query_text_with_outcome()` calls it and increments outcome counters. |
| `main.py::_evaluate` | `scorer.py` | overall + per-language scoring path | ✓ WIRED | `from scorer import scorer`; called for primary language and multilingual loop. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|---|---|---|---|---|
| `src/clef_retrieval/pipeline.py` | `evidence` / `query_text` | `service.extract_tweet_evidence(tweet_text)` then gate + `build_embedding_input(...)` | Yes (runtime Gemini extraction or deterministic fallback to tweet text) | ✓ FLOWING |
| `src/clef_retrieval/pipeline.py` | weighted ranking terms | Evidence fields + metadata row fields (`title`, `authors`, `abstract`, terms) | Yes (computed dynamic scoring per candidate) | ✓ FLOWING |
| `main.py` | `per_language_scores` | Reads cached prediction JSONL and calls `scorer(...)` per language | Yes (`scorer.py` loads dataset and computes MRR) | ✓ FLOWING |
| `main.py` | `extraction_outcomes` | `_build_query_text_with_outcome()` per query | Yes (counts real per-row outcomes; no static placeholder) | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Phase 1 behavior tests pass | `uv run pytest -q tests/test_pipeline.py tests/test_signal_weighting.py tests/test_cli.py tests/test_language_metrics_reporting.py tests/test_cost_transparency.py tests/test_query_policy.py tests/test_subset_policy.py` | `40 passed` | ✓ PASS |
| CLI exposes multilingual metrics controls | `python main.py evaluate --help` (covered in tests and parser) | Flag `--multilingual-metrics` present in parser and tests | ✓ PASS |
| Cost guardrail behavior | `tests/test_cost_transparency.py` | Guardrail warning/raise behavior validated | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| QRY-01 | 01-01, 01-02 | Extract structured tweet evidence with measurable parse-success rate | ✓ SATISFIED | Extraction metadata outcomes in pipeline + CLI telemetry counters; tests in `test_pipeline.py`. |
| QRY-02 | 01-02, 01-03 | Log/report extraction success/fallback rates | ✓ SATISFIED | `main._predict()` prints accepted/rejected/fallback outcomes and accepted rate. |
| QRY-03 | 01-02 | Use title fragments/authors in retrieval/ranking | ✓ SATISFIED | `weighted_signal_rerank()` prioritizes title+author terms with highest configured weight. |
| QRY-04 | 01-01, 01-03 | Language-aware handling for en/de/fr and per-language MRR reporting | ✓ SATISFIED | Language normalization helper + multilingual metrics reporting in `_evaluate()`. |
| QRY-05 | 01-02 | Enforce negative constraints to reduce false positives | ✓ SATISFIED | Negative term penalty in weighted scoring + explicit test coverage. |
| EVAL-01 | 01-03 | Evaluate from cached predictions unless recompute requested | ✓ SATISFIED | Cached path in `_evaluate()` and test asserting no recomputation. |
| EVAL-02 | 01-03 | Record/report overall + per-language metrics | ✓ SATISFIED | `_evaluate()` prints per-language and computed overall multilingual MRR. |
| OPS-01 | 01-01, 01-03 | Batch controls suitable for iterative experiments | ✓ SATISFIED | `query_batch_size`, subset limit/seed controls in CLI and internal selection helpers. |
| OPS-02 | 01-03 | Keep model/API usage transparent in CLI output | ✓ SATISFIED | `_estimate_query_api_usage()` + printed usage estimate and extraction mode/call counts. |
| OPS-03 | 01-03 | Keep practical cost bounds unless explicitly overridden | ✓ SATISFIED | `_enforce_cost_guardrail()` returns warning or raises unless `--allow-cost-overrun`. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---:|---|---|---|
| `main.py` | 92, 105, 202, 359 | Empty list/dict initializations | ℹ️ Info | Legitimate accumulators/state initialization; later populated through real code paths. |
| `src/clef_retrieval/query_policy.py` | 58 | `return []` edge case | ℹ️ Info | Correct behavior for invalid subset limits; not a user-facing stub. |
| `src/clef_retrieval/pipeline.py` | 65 | `return [], [], [], [], language` | ℹ️ Info | Controlled branch for non-strict mode; current config uses strict mode. |

No blocker or warning-level stubs detected in verified phase artifacts.

### Human Verification Required

None required for phase acceptance.  
(Optional manual check: run full predict/evaluate with real `GEMINI_API_KEY` to validate external API latency/cost assumptions in live conditions.)

### Gaps Summary

No blocking gaps found. All Phase 1 success criteria and mapped requirements are implemented, wired, and test-backed.

---

_Verified: 2026-03-31T00:00:32Z_  
_Verifier: the agent (gsd-verifier)_
