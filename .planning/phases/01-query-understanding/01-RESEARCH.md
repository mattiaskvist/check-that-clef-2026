# Phase 1: Query Understanding - Research

**Researched:** 2026-03-31  
**Domain:** Multilingual query understanding for scientific source retrieval (en/de/fr)  
**Confidence:** MEDIUM-HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
### Extraction strategy
- **D-01:** Use structured Gemini extraction for all queries as the default path in Phase 1 (quality-first).
- **D-02:** Apply a strict extraction quality gate before using structured fields in ranking; require parsed evidence plus minimum non-empty key fields, otherwise fallback.

### Signal weighting
- **D-03:** Prioritize signals in this order: title + author highest, then method/finding terms, then generic keywords.
- **D-04:** Use language-aware normalization per `en`/`de`/`fr` before applying weights.

### Experiment protocol
- **D-05:** Use subset-first experiment loops for iterative changes, then promote winning variants to full dev validation.
- **D-06:** Use fixed seeded subsets for reproducibility across experiments.
- **D-07:** Promotion gate to full runs: subset must improve while no language regresses.

### the agent's Discretion
- Exact key-field threshold definitions for the strict extraction gate (as long as they satisfy D-02 intent).
- Concrete weighting constants as long as ordering from D-03 is preserved.
- Exact subset sizes per language and seed-management mechanics, within D-05 to D-07.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| QRY-01 | Extract structured tweet evidence with measurable parse-success rate | Quality-gate pattern, extraction telemetry counters, schema-first parsing using existing `TweetEvidence` and `GeminiService.extract_tweet_evidence()` |
| QRY-02 | Log/report extraction success/fallback rates | Add explicit outcome categories in predict/evaluate loop (`parsed+accepted`, `parsed+rejected`, `error->fallback`) and aggregate in CLI summary |
| QRY-03 | Use extracted title fragments/authors in retrieval/ranking signals | Weighted rerank design in `pipeline.py` with title+author highest priority (D-03) |
| QRY-04 | Language-aware handling for en/de/fr + per-language MRR@5 | Per-language normalization and mandatory per-language score reporting in evaluation artifacts |
| QRY-05 | Enforce negative constraints | Candidate filtering/penalty stage before final top-5 clipping |
| EVAL-01 | Evaluate from cached predictions unless recompute requested | Existing `_evaluate()` cached path in `main.py` already supports this; preserve and extend reporting |
| EVAL-02 | Record baseline + experiment metrics (overall + per-language) | Add experiment log artifact (JSON/JSONL/CSV) with run metadata, seed, subset/full mode, per-language MRR |
| OPS-01 | Batch controls for iterative experiments | Existing `--query-batch-size`, `--limit`, and subset-first workflow; formalize seeded subset flags |
| OPS-02 | Transparent query-time model/API usage in CLI output | Print model names and extraction mode already exists; extend with extraction pass/fallback counts and estimated API calls |
| OPS-03 | Practical cost bounds unless overridden | Add budget guardrails (warnings/abort threshold) using query count × extraction+embedding calls |
</phase_requirements>

## Summary

Phase 1 should be implemented as a strict **query-understanding quality layer** on top of the current pipeline, not a rewrite. The repository already has the right primitives: structured schema (`TweetEvidence`), extraction client (`GeminiService.extract_tweet_evidence`), query construction hook (`build_query_embedding_text`), cached evaluation path, and CLI batch controls. What is missing is strict gating, explicit multilingual normalization, signal-aware ranking logic, and observability.

The highest-value plan is: keep Gemini structured extraction as default (D-01), gate extracted output before use (D-02), and apply deterministic weighting where title+author outrank all other signals (D-03), with language-aware normalization (D-04). Couple this with reproducible subset-first experiments (D-05 to D-07), and metrics logging that makes regressions visible by language and by extraction outcome.

**Primary recommendation:** Implement a gated, language-aware evidence scoring/ranking layer in `pipeline.py` plus extraction/metrics instrumentation in `main.py`, while preserving cached evaluation flow and fallback reliability.

## Project Constraints (from copilot-instructions.md)

- Use the existing retrieval-research scope only (no UI/deployment expansion).
- Stay aligned with chosen direction: Gemini Embeddings 2 + LLM-assisted query understanding.
- Keep evaluation target as dev MRR@5 vs in-repo baseline.
- Do not introduce new external datasets; remain within current CLEF splits.
- Maintain cost-awareness for iterative runs (initial budget-sensitive operation).
- Follow Python project standards in repo: Python `>=3.14`, `uv` workflow, `pytest`, `ruff`, Pydantic schemas.
- Preserve existing CLI-centric experimentation model and cache-based workflow.
- Follow naming/style conventions (snake_case modules/functions, typed signatures where used, schema-first validation).

## Standard Stack

### Core
| Library | Version (in repo) | Latest verified | Purpose | Why Standard |
|---------|--------------------|-----------------|---------|--------------|
| google-genai | `>=1.68.0` | 1.69.0 (2026-03-28) | Gemini extraction + embeddings | Already integrated and central to locked decisions |
| pydantic | `>=2.12.5` | 2.12.5 (2025-11-26) | Structured evidence/config validation | Existing schema-driven design in codebase |
| numpy | `>=2.4.3` | 2.4.4 (2026-03-29) | Dense similarity and ranking arrays | Core retrieval operations already depend on it |

### Supporting
| Library | Version (in repo) | Latest verified | Purpose | When to Use |
|---------|--------------------|-----------------|---------|-------------|
| datasets | `>=4.8.4` | 4.8.4 (2026-03-23) | CLEF split loading for eval/scoring | All predict/evaluate paths |
| tqdm | `>=4.67.1` | 4.67.3 (2026-02-03) | Batch progress + CLI transparency | Required for OPS-02 visibility |
| python-dotenv | `>=1.2.2` | 1.2.2 (2026-03-01) | Local key loading for CLI | Predict/evaluate/build-index entrypoints |
| pytest | `>=8.4.2` | 9.0.2 (2025-12-06) | Validation architecture | Unit/integration behavior checks |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Gemini structured extraction | Regex/heuristic parsing | Lower API cost but fails multilingual/noisy scientific tweets; conflicts with D-01 quality-first |
| Weighted signal rerank in pipeline | Raw tweet embedding only | Simpler but does not satisfy QRY-03/QRY-05 and underuses existing schema |

**Installation:**
```bash
uv sync
```

**Version verification:** Verified via PyPI registry API (`https://pypi.org/pypi/<package>/json`) on 2026-03-31.

## Architecture Patterns

### Recommended Project Structure
```text
src/clef_retrieval/
├── schemas.py              # Evidence models (TweetEvidence)
├── config.py               # Retrieval/query gating + weighting config
├── gemini_client.py        # Structured extraction + embeddings
├── pipeline.py             # Gate, weighting, negative constraints, rerank composition
└── evaluation.py           # Metrics helpers (plus CLI-level reporting in main.py)
main.py                     # predict/evaluate orchestration and telemetry output
tests/                      # unit + CLI behavior regression tests
```

### Pattern 1: Strict extraction quality gate before structured use
**What:** Accept structured evidence only when parsed and meeting key-field thresholds; otherwise fallback to raw tweet.
**When to use:** Every query before embedding/ranking.
**Example:**
```python
# Source: src/clef_retrieval/pipeline.py + src/clef_retrieval/schemas.py
evidence = service.extract_tweet_evidence(tweet_text)
if not isinstance(evidence, TweetEvidence):
    return tweet_text  # fallback
```

### Pattern 2: Weighted multi-signal ranking (title+author > method/finding > keywords)
**What:** Deterministic score composition from extracted fields.
**When to use:** Reordering dense top-K candidates.
**Example:**
```python
# Source anchor: src/clef_retrieval/pipeline.py::_lexical_rerank (to be replaced/extended)
# target formula direction (D-03):
# score = 5*title_author + 3*method_finding + 1*keywords - penalty(negative_constraints)
```

### Pattern 3: Subset-first reproducible experiment loop
**What:** Evaluate changes on fixed seeded subsets per language, then promote to full dev only if no language regresses.
**When to use:** Every iterative tuning cycle.
**Example:**
```bash
# Source anchor: main.py flags (--limit, --query-batch-size, cached evaluate path)
uv run main.py evaluate --lang en --split dev --limit 100
```

### Anti-Patterns to Avoid
- **Ungated structured use:** Consuming partially valid extraction without threshold checks causes noisy ranking regressions.
- **Aggregate-only reporting:** Overall MRR can improve while one language regresses (violates D-07 intent).
- **Implicit fallback:** Silent fallback without counters makes QRY-02 impossible to verify.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON schema parsing | Manual dict parsing/if-chains | Pydantic `TweetEvidence` | Existing typed validation and parse-failure behavior |
| Retry/backoff for Gemini errors | Ad-hoc loops | Existing `GeminiService.embed_texts()` retry strategy | Already handles 429/resource exhaustion with exponential backoff |
| MRR metric computation | Custom variant per experiment | Existing `scorer.py` | Prevents metric drift and keeps baseline comparability |
| Top-5 shape/padding | One-off padding logic in multiple places | `ensure_top5()` + existing pipeline flow | Avoids inconsistent output shape bugs |

**Key insight:** Most Phase-1 risk is not missing capability; it is inconsistent policy application. Reuse existing primitives and centralize gate/weight/report logic.

## Common Pitfalls

### Pitfall 1: False confidence from parsed-but-weak extraction
**What goes wrong:** Parsed JSON exists, but key fields are empty/noisy; ranking quality degrades.
**Why it happens:** Parse success is treated as quality success.
**How to avoid:** Separate `parse_success` from `gate_pass` and enforce minimum key-field thresholds.
**Warning signs:** Rising parse rate without MRR improvement; increased fallback-like behavior in output quality.

### Pitfall 2: Language regression hidden by macro averages
**What goes wrong:** Overall MRR improves while one of en/de/fr degrades.
**Why it happens:** Reporting only aggregate score.
**How to avoid:** Always log/report per-language MRR in every experiment.
**Warning signs:** Inconsistent behavior reports by language; surprising full-dev results after subset wins.

### Pitfall 3: Negative constraints extracted but ignored
**What goes wrong:** Obvious false positives remain in top-5 despite explicit exclusions.
**Why it happens:** Constraints are in schema but never applied in ranking/filtering.
**How to avoid:** Apply hard filter or score penalty stage before `clip_top5`.
**Warning signs:** Predicted papers clearly match prohibited terms from query.

## Code Examples

Verified patterns from repository sources:

### Safe extraction fallback
```python
# Source: src/clef_retrieval/pipeline.py
def build_query_embedding_text(tweet_text: str, service: GeminiService) -> str:
    try:
        evidence = service.extract_tweet_evidence(tweet_text)
        if not isinstance(evidence, TweetEvidence):
            raise QueryExtractionError("Gemini response did not parse as TweetEvidence")
        return build_embedding_input(evidence, tweet_text)
    except (...):
        return tweet_text
```

### Cached evaluation without recompute
```python
# Source: main.py::_evaluate
if prediction_path.exists() and not args.recompute:
    print(f"Using cached predictions from {prediction_path}")
    rows = _read_predictions(prediction_path)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Raw-tweet-only query representation | Structured extraction available with fallback | Already in current repo | Enables Phase-1 quality gating and weighted signals |
| Opaque extraction behavior | Explicit extraction outcome telemetry | Phase 1 target | Enables QRY-02 regression detection |
| Lexical overlap rerank only | Signal-weighted multilingual rerank using extracted fields | Phase 1 target | Addresses QRY-03/QRY-05 with current architecture |

**Deprecated/outdated:**
- Relying on aggregate-only score comparisons for iteration decisions; replace with per-language + extraction outcome metrics.

## Open Questions

1. **Exact gate thresholds for D-02**
   - What we know: Must require parsed evidence + non-empty key fields.
   - What's unclear: Best minimums (e.g., title/author count, confidence surrogate).
   - Recommendation: Start conservative (high precision), tune via seeded subset ablations.

2. **Negative constraints policy**
   - What we know: Must be enforced (QRY-05).
   - What's unclear: Hard filter vs soft penalty in borderline cases.
   - Recommendation: Implement hard exclusion for explicit negations first; add penalty mode behind config if recall loss appears.

3. **Experiment artifact format**
   - What we know: Need reproducible baseline vs experiment tracking (EVAL-02).
   - What's unclear: JSONL vs CSV vs markdown summary as source of truth.
   - Recommendation: JSONL run-log as canonical + concise CLI table output.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| python3 | CLI/pipeline/test execution | ✓ | 3.14.0 | — |
| uv | Dependency + run workflow | ✓ | 0.9.2 | `python -m pip` (lower alignment with project conventions) |
| git | Baseline/experiment traceability | ✓ | 2.50.1 | — |
| curl | Registry/docs checks | ✓ | 8.7.1 | — |
| node/npm | GSD helper tooling | ✓ | v22.22.0 / 11.11.1 | — |
| GEMINI_API_KEY | build-index/predict/evaluate with extraction/embedding | Not verifiable from static audit | — | Required runtime secret; cannot run full pipeline without it |
| HuggingFace auth/session | Dataset/scoring access | Not verifiable from static audit | — | Must authenticate per README instructions |

**Missing dependencies with no fallback:**
- Valid Gemini API credentials at runtime for extraction/embedding paths.

**Missing dependencies with fallback:**
- None confirmed missing from local toolchain audit.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (repo constraint `>=8.4.2`, latest verified 9.0.2) |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options] pythonpath = ["src"]`) |
| Quick run command | `uv run pytest -q tests/test_pipeline.py tests/test_cli.py` |
| Full suite command | `uv run pytest -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| QRY-01 | Structured extraction + parse success reporting | unit/integration | `uv run pytest -q tests/test_pipeline.py` | ❌ Wave 0 (needs new assertions/counters) |
| QRY-02 | Fallback/success rate telemetry | unit/CLI | `uv run pytest -q tests/test_cli.py` | ❌ Wave 0 |
| QRY-03 | Title/author signals affect ranking | unit | `uv run pytest -q tests/test_pipeline.py` | ❌ Wave 0 |
| QRY-04 | Language-aware handling + per-language metrics | integration | `uv run pytest -q tests/test_cli.py` | ❌ Wave 0 |
| QRY-05 | Negative constraints enforcement | unit | `uv run pytest -q tests/test_pipeline.py` | ❌ Wave 0 |
| EVAL-01 | Cached eval reuse without recompute | integration | `uv run pytest -q tests/test_cli.py::test_evaluate_uses_cached_predictions_without_recomputing` | ✅ |
| EVAL-02 | Baseline/experiment metric recording | integration | `uv run pytest -q tests/test_cli.py` | ❌ Wave 0 |
| OPS-01 | Batch controls usable for iteration | integration | `uv run pytest -q tests/test_cli.py` | ✅ (flags exist), ❌ (subset-seed policy checks) |
| OPS-02 | Transparent model/API usage in CLI output | integration | `uv run pytest -q tests/test_cli.py` | ❌ Wave 0 |
| OPS-03 | Cost guardrails/warnings | unit/integration | `uv run pytest -q tests/test_cli.py` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest -q tests/test_pipeline.py tests/test_cli.py`
- **Per wave merge:** `uv run pytest -q`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_query_understanding_gate.py` — gate-pass vs fallback classification, threshold behavior (QRY-01/QRY-02)
- [ ] `tests/test_signal_weighting.py` — title/author priority ordering and negative constraints (QRY-03/QRY-05)
- [ ] `tests/test_language_metrics_reporting.py` — per-language reporting + no-regression promotion logic (QRY-04/EVAL-02)
- [ ] `tests/test_cost_transparency.py` — API usage/cost visibility and guardrails (OPS-02/OPS-03)

## Sources

### Primary (HIGH confidence)
- Repository code (local): `main.py`, `src/clef_retrieval/pipeline.py`, `src/clef_retrieval/gemini_client.py`, `src/clef_retrieval/config.py`, `src/clef_retrieval/schemas.py`
- Repository planning artifacts: `.planning/phases/01-query-understanding/01-CONTEXT.md`, `.planning/REQUIREMENTS.md`, `.planning/STATE.md`, `.planning/ROADMAP.md`, `.planning/PROJECT.md`
- Repository tests: `tests/test_pipeline.py`, `tests/test_cli.py`, `tests/test_main_progress.py`, `tests/test_gemini_client.py`

### Secondary (MEDIUM confidence)
- PyPI official registry JSON endpoints for current package versions and upload timestamps (queried 2026-03-31)

### Tertiary (LOW confidence)
- Prior internal research notes in `.planning/research/*.md` (useful directionally, some content dated and partially unverified against external docs in this run)

## Metadata

**Confidence breakdown:**
- Standard stack: **HIGH** — directly verified from repo + PyPI current versions
- Architecture: **MEDIUM-HIGH** — based on concrete existing code paths and locked decisions
- Pitfalls: **MEDIUM** — strongly grounded in current design but full impact requires experiment data

**Research date:** 2026-03-31  
**Valid until:** 2026-04-30 (or earlier if Gemini model defaults/API behavior changes)
