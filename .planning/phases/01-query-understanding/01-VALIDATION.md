---
phase: 1
slug: query-understanding
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-31
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `uv run pytest -q tests/test_pipeline.py tests/test_cli.py` |
| **Full suite command** | `uv run pytest -q` |
| **Estimated runtime** | ~20-60 seconds (suite-dependent) |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -q tests/test_pipeline.py tests/test_cli.py`
- **After every plan wave:** Run `uv run pytest -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | QRY-01, QRY-04, OPS-01 | unit | `uv run pytest -q tests/test_config.py` | ✅ | ⬜ pending |
| 01-01-02 | 01 | 1 | QRY-01, QRY-04, OPS-01 | unit | `uv run pytest -q tests/test_query_policy.py tests/test_subset_policy.py` | ❌ Wave 0 | ⬜ pending |
| 01-01-03 | 01 | 1 | QRY-01, QRY-04, OPS-01 | unit | `uv run pytest -q tests/test_query_policy.py tests/test_subset_policy.py` | ❌ Wave 0 | ⬜ pending |
| 01-02-01 | 02 | 2 | QRY-01, QRY-02, QRY-03, QRY-05 | unit | `uv run pytest -q tests/test_pipeline.py tests/test_signal_weighting.py` | ❌ Wave 0 | ⬜ pending |
| 01-02-02 | 02 | 2 | QRY-01, QRY-02, QRY-03, QRY-05 | unit | `uv run pytest -q tests/test_pipeline.py tests/test_signal_weighting.py` | ❌ Wave 0 | ⬜ pending |
| 01-02-03 | 02 | 2 | QRY-02 | unit | `uv run pytest -q tests/test_pipeline.py` | ✅ | ⬜ pending |
| 01-03-01 | 03 | 3 | QRY-02, QRY-04, EVAL-02, OPS-02, OPS-03 | integration | `uv run pytest -q tests/test_cli.py tests/test_language_metrics_reporting.py tests/test_cost_transparency.py` | ❌ Wave 0 | ⬜ pending |
| 01-03-02 | 03 | 3 | QRY-02, QRY-04, EVAL-01, EVAL-02, OPS-01 | integration | `uv run pytest -q tests/test_cli.py tests/test_language_metrics_reporting.py` | ❌ Wave 0 | ⬜ pending |
| 01-03-03 | 03 | 3 | OPS-02, OPS-03 | unit/integration | `uv run pytest -q tests/test_cost_transparency.py tests/test_cli.py` | ❌ Wave 0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_query_policy.py` — strict extraction gate + normalization + promotion helper checks
- [ ] `tests/test_subset_policy.py` — deterministic seeded subset behavior checks
- [ ] `tests/test_signal_weighting.py` — ranking-order and negative-constraint behavior checks
- [ ] `tests/test_language_metrics_reporting.py` — per-language metrics and no-regression gate checks
- [ ] `tests/test_cost_transparency.py` — model/API usage + cost guardrail reporting checks

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CLI runtime usability for long prediction loops | OPS-01 | Throughput/UX perception not fully captured by unit tests | Run `uv run python main.py predict --lang en --split dev --limit 200 --query-batch-size 32` and verify progress/summary readability |
| Cost warning usefulness for real API runs | OPS-03 | Live billing/quota context unavailable in CI | Run predict/evaluate with and without override flags under real API key and confirm warning/abort behavior |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s for quick loop
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
