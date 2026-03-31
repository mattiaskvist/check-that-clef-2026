---
phase: 2
slug: semantic-reranking
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-31
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `uv run pytest -q tests/test_pipeline.py tests/test_signal_weighting.py tests/test_language_metrics_reporting.py` |
| **Full suite command** | `uv run pytest -q` |
| **Estimated runtime** | ~25-70 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -q tests/test_pipeline.py tests/test_signal_weighting.py tests/test_language_metrics_reporting.py`
- **After every plan wave:** Run `uv run pytest -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | RNK-01 | unit | `uv run pytest -q tests/test_reranker_backends.py` | ❌ Wave 0 | ⬜ pending |
| 02-01-02 | 01 | 1 | RNK-01, RNK-02 | unit/integration | `uv run pytest -q tests/test_pipeline_semantic_rerank.py` | ❌ Wave 0 | ⬜ pending |
| 02-01-03 | 01 | 1 | RNK-01, RNK-02 | unit | `uv run pytest -q tests/test_pipeline_semantic_rerank.py tests/test_signal_weighting.py` | ❌ Wave 0 | ⬜ pending |
| 02-02-01 | 02 | 2 | EVAL-03 | integration | `uv run pytest -q tests/test_stage_diagnostics.py` | ❌ Wave 0 | ⬜ pending |
| 02-02-02 | 02 | 2 | RNK-02, EVAL-03 | integration | `uv run pytest -q tests/test_stage_diagnostics.py tests/test_language_metrics_reporting.py` | ❌ Wave 0 | ⬜ pending |
| 02-02-03 | 02 | 2 | RNK-02, EVAL-03 | integration | `uv run pytest -q tests/test_cli_semantic_flags.py tests/test_cost_transparency.py` | ❌ Wave 0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_reranker_backends.py` — backend selection contract + Jina default + BGE compatibility checks
- [ ] `tests/test_pipeline_semantic_rerank.py` — semantic-primary ordering and weighted tie-break determinism checks
- [ ] `tests/test_stage_diagnostics.py` — Recall@K, uplift delta, and latency/throughput diagnostics checks
- [ ] `tests/test_cli_semantic_flags.py` — reranker backend/depth flag behavior and diagnostics toggle checks
- [ ] `uv add sentence-transformers FlagEmbedding einops` (or documented raw-transformers fallback path with tests)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| End-to-end semantic reranker runtime sanity on real models | RNK-01, RNK-02 | Model download/runtime characteristics depend on local hardware/network and cannot be fully mocked in CI | Run `uv run python main.py predict --lang en --split dev --limit 50 --reranker-backend jina_v2 --rerank-top-k 50` and verify completion time + non-empty reranked outputs |
| Diagnostic usefulness for bottleneck localization | EVAL-03 | Human judgment needed to confirm reported diagnostics are actionable | Run evaluate with diagnostics enabled and confirm output clearly distinguishes Recall@K misses vs rerank uplift effects |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s for quick loop
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
