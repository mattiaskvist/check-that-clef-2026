# Phase 1: Query Understanding - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-31
**Phase:** 01-query-understanding
**Areas discussed:** Extraction strategy, Signal weighting, Experiment protocol

---

## Extraction strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Structured Gemini extraction for all queries | Prioritize retrieval quality by always extracting structured evidence | ✓ |
| Structured extraction only on subset/debug runs, skip on full runs | Reduce latency/cost in full runs at potential quality loss | |
| Two-pass routing | Use raw-first, structured extraction only for uncertain queries | |

**User's choice:** Structured Gemini extraction for all queries.
**Notes:** Quality-first default was preferred.

| Option | Description | Selected |
|--------|-------------|----------|
| Strict gate | Require parsed evidence and minimum non-empty key fields, else fallback | ✓ |
| Moderate gate | Accept sparse parsed evidence | |
| Lenient gate | Use parsed evidence whenever JSON parses | |

**User's choice:** Strict extraction gate.
**Notes:** Fallback remains required for low-quality extractions.

---

## Signal weighting

| Option | Description | Selected |
|--------|-------------|----------|
| Title+author first | Highest weight on title/author, then method/finding, then generic keywords | ✓ |
| Method/finding first | Prioritize method/finding terms over title/author | |
| Balanced | Equal weight across all extracted fields | |

**User's choice:** Title+author first.
**Notes:** Chosen to improve disambiguation and direct source matching.

| Option | Description | Selected |
|--------|-------------|----------|
| Language-aware normalization | Normalize separately for en/de/fr before weighting | ✓ |
| Single normalization | One normalization path for all languages | |

**User's choice:** Language-aware normalization.
**Notes:** Explicit multilingual robustness requirement.

---

## Experiment protocol

| Option | Description | Selected |
|--------|-------------|----------|
| Subset-first loop | Iterate on subsets, then validate on full dev | ✓ |
| Full-dev every run | No subset loop | |
| Per-language subsets first | Language-first staged loop | |

**User's choice:** Subset-first loop.
**Notes:** Aligned with known prediction bottleneck from query-time generation.

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed seeded subsets | Reproducible subsets across runs | ✓ |
| Ad-hoc subsets | Flexible subsets per run | |

**User's choice:** Fixed seeded subsets.
**Notes:** Reproducibility prioritized for fair comparisons.

| Option | Description | Selected |
|--------|-------------|----------|
| No language regressions gate | Promote only if improvement and no per-language regression | ✓ |
| Average-only gate | Promote on overall average improvement | |
| Every N runs | Promote periodically regardless of outcome | |

**User's choice:** No-language-regression promotion gate.
**Notes:** Prevents masking regressions in specific languages.

---

## the agent's Discretion

- Exact numeric thresholds for strict extraction acceptance.
- Exact weighting constants (preserving user-selected order).
- Exact subset sizes and seed policy implementation details.

## Deferred Ideas

None raised during this discussion.
