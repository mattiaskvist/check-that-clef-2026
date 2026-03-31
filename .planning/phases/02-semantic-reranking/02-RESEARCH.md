# Phase 2: semantic-reranking - Research

**Researched:** 2026-03-31  
**Domain:** Multilingual cross-encoder reranking for dense retrieval candidates  
**Confidence:** MEDIUM

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
### Reranker model strategy
- **D-01:** Implement configurable support for both Jina Reranker v2 and BGE v2-m3.
- **D-02:** Default reranker backend for Phase 2 experiments should be Jina Reranker v2.

### Candidate handoff and rerank depth
- **D-03:** Rerank the top **50** dense-retrieval candidates per query (`rerank_top_k=50`).
- **D-04:** Keep rerank depth as an explicit config/CLI-controlled parameter so experiments can compare cost/quality trade-offs without code changes.

### Score combination policy
- **D-05:** Use semantic reranker score as primary ordering signal.
- **D-06:** Use Phase 1 weighted signal score only as a tie-breaker (not as a blended primary score).

### Diagnostics and evaluation scope
- **D-07:** Include stage-level dense retrieval Recall@K diagnostics.
- **D-08:** Report semantic reranked MRR@5 and uplift delta (overall + per-language) against baseline.
- **D-09:** Report reranker latency/throughput diagnostics in evaluation output.

### the agent's Discretion
- Exact tie-break implementation mechanics (stable sort strategy, epsilon threshold) as long as D-05/D-06 hold.
- Concrete config field names and defaults for reranker backend selection and rerank depth.
- Internal diagnostics representation format (stdout table vs structured JSON lines) as long as D-07 to D-09 remain visible and testable.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

## Project Constraints (from copilot-instructions.md)

- Task scope is CLEF 2026 CheckThat Task 1 retrieval only.
- Keep model direction consistent with Gemini Embeddings 2 + LLM-assisted query understanding.
- Evaluate progress by dev MRR@5 against current in-repo baseline.
- Stay within existing CLEF dataset boundaries (no new external datasets).
- Preserve cost-awareness and practical iteration speed.
- Use subset-first experimentation before full-split validation.
- Follow existing Python conventions (snake_case, typed functions, Pydantic config-first patterns, pytest-based validation).
- Work within GSD workflow expectations (phase artifacts and planning context alignment).

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RNK-01 | System replaces or augments lexical-only reranking with semantic reranking over top-K dense candidates. | Recommends `SemanticReranker` interface + backend adapters (Jina default, BGE optional), integrated into `rank_from_query_embedding()` after dense retrieval. |
| RNK-02 | System reranking strategy preserves or improves multilingual behavior across en/de/fr. | Recommends multilingual-capable model choices (Jina v2 multilingual / BGE v2-m3), per-language uplift reporting, and no-regression promotion checks reusing Phase 1 policy. |
| EVAL-03 | System provides stage-level diagnostics (retrieval recall-at-K and reranking uplift indicators) to localize bottlenecks. | Recommends explicit diagnostics payload: Recall@50, MRR@5 baseline vs reranked, uplift deltas, and reranker latency/throughput metrics in CLI/evaluation output. |
</phase_requirements>

## Summary

Phase 2 should introduce a true cross-encoder reranking stage at the existing `rank_from_query_embedding()` boundary, not a replacement of dense retrieval. The dense stage remains candidate generator (`top_k`), and semantic reranking is applied to a controlled depth (`rerank_top_k=50`) with deterministic tie-breaking using the existing Phase 1 weighted score only when semantic scores are effectively equal. This preserves prior retrieval stability while meeting the semantic-first policy in D-05/D-06.

The current codebase is already well-positioned for this integration: `retrieve_top_pubkeys()` is deterministic, `RerankResult` already exists, `main._evaluate()` already supports multilingual reporting and promotion gating, and cost/telemetry surfaces already exist. The biggest engineering risk is runtime/ops: local environment currently lacks `sentence-transformers` and `FlagEmbedding`, and Jina v2 uses `trust_remote_code=True` for best support. Planning must include dependency setup, backend abstraction, and diagnostics wiring as first-class tasks.

To meet RNK-02/EVAL-03 credibly, do not rely on aggregate MRR only. Add stage-level metrics that separate candidate-recall failure (retrieval miss) from ordering failure (rerank miss). This makes Phase 2 decision-making actionable and avoids wasted optimization in the wrong stage.

**Primary recommendation:** Implement a backend-swappable semantic reranker adapter (Jina default, BGE optional) invoked after dense retrieval over top-50, with semantic-primary ordering + deterministic weighted-signal tie-break, and extend `evaluate` to emit retrieval recall + reranking uplift + latency diagnostics.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | 5.4.0 (PyPI, 2026-03-27) | Load HF reranker models (including sequence classification rerankers) | Official HF runtime used by both Jina/BGE model cards and current project dependencies |
| torch | 2.11.0 (PyPI, 2026-03-23) | Inference runtime for reranker models | Required execution backend for cross-encoders |
| sentence-transformers | 5.3.0 (PyPI, 2026-03-12) | CrossEncoder abstraction for query-doc scoring | Standard retrieve→rerank pattern docs and mature batching helpers |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| FlagEmbedding | 1.3.5 (PyPI, 2025-05-28) | Native BGE reranker wrapper (`FlagReranker`) | Use when BGE backend selected and you want BGE-maintained scoring API |
| einops | 0.8.2 (PyPI, 2026-01-26) | Required by some Jina model usage examples | Add when Jina backend path requires it in local env |
| huggingface-hub | 1.8.0 (PyPI, 2026-03-25) | Model download/auth tooling | Use for cached model management/offline prefetch |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| sentence-transformers CrossEncoder | Raw `transformers` with manual tokenization/scoring | More control, but more hand-rolled batching/padding/error handling |
| FlagEmbedding for BGE | sentence-transformers CrossEncoder over BGE model | Simpler dependency graph, but may miss BGE-specific optimizations |
| Local Jina HF inference | Jina hosted reranker API | Faster onboarding, but adds external paid dependency and network variance |

**Installation:**
```bash
uv add sentence-transformers FlagEmbedding einops
```

**Version verification:**  
Verified via PyPI JSON API on 2026-03-31:
- `transformers==5.4.0` (uploaded 2026-03-27)
- `torch==2.11.0` (uploaded 2026-03-23)
- `sentence-transformers==5.3.0` (uploaded 2026-03-12)
- `FlagEmbedding==1.3.5` (uploaded 2025-05-28)

## Architecture Patterns

### Recommended Project Structure
```text
src/clef_retrieval/
├── reranker.py              # backend-agnostic protocol + adapters + scoring helpers
├── pipeline.py              # dense retrieval + semantic rerank orchestration
├── config.py                # reranker backend/depth/device/tie-break fields
└── diagnostics.py           # stage metrics helpers (recall@K, uplift, latency stats)
tests/
├── test_reranker_semantic.py
├── test_pipeline_semantic_rerank.py
└── test_stage_diagnostics.py
```

### Pattern 1: Retrieve → Semantic Rerank (fixed depth)
**What:** Keep dense retrieval as recall stage, rerank only top-N candidates with cross-encoder.  
**When to use:** Always for Phase 2 baseline path (`rerank_top_k=50`).  
**Example:**
```python
# Source: existing integration point (src/clef_retrieval/pipeline.py::rank_from_query_embedding)
candidates = retrieve_top_pubkeys(query_embedding, paper_embeddings, pubkeys, k=cfg.top_k)
rerank_candidates = candidates[: cfg.rerank_top_k]
scored = semantic_reranker.score(query_text, rerank_candidates, metadata_by_pubkey)
ordered = sort_semantic_primary_with_weighted_tiebreak(scored, weighted_scores)
top5 = ensure_top5(ordered[:5])
```

### Pattern 2: Backend Adapter Interface
**What:** One interface, two backends (`jina_v2`, `bge_v2_m3`) selected by config/CLI.  
**When to use:** Required by D-01/D-02 to avoid code forks.  
**Example:**
```python
class SemanticReranker(Protocol):
    def score(self, query: str, candidates: list[str], metadata_by_pubkey: dict[str, dict]) -> list[tuple[str, float]]: ...
```

### Pattern 3: Deterministic Semantic-Primary Tie-break
**What:** Sort by semantic score descending; for near-equal scores (`abs(a-b) <= epsilon`), apply Phase 1 weighted score; final fallback = original candidate order.  
**When to use:** Every rerank output to preserve reproducibility and D-05/D-06.  

### Anti-Patterns to Avoid
- **Blended primary score (semantic + lexical weighted sum):** violates D-05 and complicates diagnostics.
- **Reranking full `top_k` blindly:** large latency cost without evidence; keep explicit `rerank_top_k`.
- **Backend-specific branches in pipeline core:** creates test explosion and unstable behavior; isolate in adapters.
- **Non-deterministic tie handling:** breaks seeded subset reproducibility and promotion gating comparability.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pair tokenization, truncation, batching | Custom tokenizer/batch loops | `sentence_transformers.CrossEncoder` or backend wrappers | Mature handling of padding/truncation/device edge cases |
| Manual model download/caching logic | Ad hoc file management | HuggingFace model resolution via transformers/huggingface-hub | Reproducible cache behavior and pinned model revisions |
| Full reranker scoring calibration | Custom post-hoc score normalization as primary metric | Rank by raw model score; use tie-break epsilon only | Ranking quality depends on order, not absolute score scale |
| Recall@K/MRR bookkeeping in-line in CLI | Duplicated ad hoc metrics code | Dedicated diagnostics helpers + existing `scorer` pipeline | Keeps verification testable and maintainable |

**Key insight:** Cross-encoder infra complexity is in tokenization/batching/runtime behavior, not just “call model and sort”; use proven libraries and keep custom code to orchestration + deterministic policies.

## Common Pitfalls

### Pitfall 1: Trust-Remote-Code Blind Spot (Jina backend)
**What goes wrong:** Jina v2 usage often requires `trust_remote_code=True`; without controlled policy this can be blocked or unsafe.  
**Why it happens:** Some reranker repos ship custom model code.  
**How to avoid:** Add explicit config gate (`allow_trust_remote_code`), default true only for approved model IDs, and document accepted repos.  
**Warning signs:** model load failures or silent backend fallback.

### Pitfall 2: Latency Blow-up from Unbounded Rerank Depth
**What goes wrong:** Cross-encoder inference cost scales with candidate count; full-depth reranking becomes too slow/costly.  
**Why it happens:** Dense `top_k` and rerank depth conflated.  
**How to avoid:** Separate `top_k` (retrieval) vs `rerank_top_k` (reranker) and default to 50 per D-03.  
**Warning signs:** Evaluate runtime spikes; low throughput in diagnostics.

### Pitfall 3: Score Semantics Misinterpreted Across Backends
**What goes wrong:** Treating scores as calibrated probabilities across Jina/BGE causes unstable thresholds.  
**Why it happens:** Different model heads/output ranges.  
**How to avoid:** Compare by rank order only; keep backend-specific score interpretation out of acceptance logic.  
**Warning signs:** same ranking quality but misleading “score drops” trigger false alarms.

### Pitfall 4: No Stage Diagnostics Means Wrong Optimization
**What goes wrong:** Team tunes reranker when root cause is retrieval miss (gold not in top-K).  
**Why it happens:** Only final MRR reported.  
**How to avoid:** Emit Recall@K before rerank + uplift delta after rerank (overall and by language).  
**Warning signs:** flat MRR with unclear failure mode.

## Code Examples

Verified patterns from official sources:

### Cross-encoder scoring with sentence-transformers
```python
# Source: Sentence-Transformers CrossEncoder docs / source
from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-v2-m3")
pairs = [[query, doc] for doc in documents]
scores = model.predict(pairs, batch_size=32)
```

### Jina reranker via transformers (model card pattern)
```python
# Source: HF model card: jinaai/jina-reranker-v2-base-multilingual
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "jinaai/jina-reranker-v2-base-multilingual",
    torch_dtype="auto",
    trust_remote_code=True,
)
scores = model.compute_score([[query, doc] for doc in documents], max_length=1024)
```

### BGE reranker via FlagEmbedding
```python
# Source: HF model card: BAAI/bge-reranker-v2-m3
from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
scores = reranker.compute_score([[query, doc] for doc in documents])
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Lexical overlap rerank on dense candidates | Multilingual cross-encoder rerank (Jina v2 / BGE v2-m3) | Current IR best-practice era; aligned with retrieve-rerank pipelines | Better precision in top ranks, especially multilingual paraphrase cases |
| Single final metric only | Stage diagnostics (Recall@K + rerank uplift + latency) | Modern evaluation practice in retrieval iteration loops | Faster bottleneck localization and safer iteration |

**Deprecated/outdated:**
- Lexical-only reranking as primary stage for multilingual semantic matching.
- Treating aggregate metric alone as sufficient for retrieval pipeline optimization.

## Open Questions

1. **Jina local runtime compatibility in this repo environment**
   - What we know: Jina model card recommends `trust_remote_code=True`, optional flash-attn for speed.
   - What's unclear: exact CPU fallback performance and whether additional deps are required in this uv environment.
   - Recommendation: add Wave 0 smoke test task loading model on CPU with 3-pair scoring.

2. **Backend default under license constraints**
   - What we know: Jina HF model card is CC-BY-NC-4.0; BGE is Apache-2.0.
   - What's unclear: any project policy constraints for non-commercial license during experimentation.
   - Recommendation: keep Jina default per D-02, but add explicit backend switch and document license note.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | runtime | ✓ | 3.14.0 | — |
| uv | package/test workflow | ✓ | 0.9.2 | pip (slower/manual) |
| torch (uv env) | reranker inference | ✓ | 2.11.0 | — |
| transformers (uv env) | model loading/tokenization | ✓ | 5.3.0 | — |
| sentence-transformers (uv env) | CrossEncoder adapter path | ✗ | — | use raw transformers path |
| FlagEmbedding (uv env) | BGE native adapter path | ✗ | — | use CrossEncoder/raw transformers |
| HuggingFace CLI auth (`hf`) | large model fetch convenience | partial | command present, no `--version` support | rely on implicit public model download |

**Missing dependencies with no fallback:**
- None blocking for implementation (raw transformers fallback exists).

**Missing dependencies with fallback:**
- `sentence-transformers` (fallback: raw transformers adapter)
- `FlagEmbedding` (fallback: BGE via transformers/CrossEncoder)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Quick run command | `uv run pytest -q tests/test_pipeline.py tests/test_reranker.py tests/test_language_metrics_reporting.py` |
| Full suite command | `uv run pytest -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RNK-01 | Semantic reranker replaces lexical-only as primary ordering on top-50 | unit + integration | `uv run pytest -q tests/test_pipeline_semantic_rerank.py::test_semantic_primary_ordering` | ❌ Wave 0 |
| RNK-02 | No per-language regression in reranked MRR@5 (en/de/fr) | integration/cli | `uv run pytest -q tests/test_stage_diagnostics.py::test_multilingual_uplift_reporting` | ❌ Wave 0 |
| EVAL-03 | Evaluate emits Recall@K + reranking uplift + latency/throughput | unit + cli | `uv run pytest -q tests/test_stage_diagnostics.py::test_stage_metrics_output` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest -q tests/test_pipeline.py tests/test_reranker.py tests/test_language_metrics_reporting.py tests/test_cost_transparency.py`
- **Per wave merge:** `uv run pytest -q`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_pipeline_semantic_rerank.py` — semantic-primary ordering + weighted tie-break determinism
- [ ] `tests/test_reranker_backends.py` — backend selection, Jina default, BGE compatibility contract
- [ ] `tests/test_stage_diagnostics.py` — Recall@K, uplift delta, reranker latency/throughput output
- [ ] `tests/test_cli_semantic_flags.py` — `--reranker-backend`, `--rerank-top-k`, diagnostics toggles
- [ ] Dependency install step in setup docs: `uv add sentence-transformers FlagEmbedding einops`

## Planner Task Breakdown Guidance

1. **Wave 0 (contracts + scaffolding)**
   - Add config fields: `reranker_backend`, `rerank_top_k`, `reranker_batch_size`, `reranker_device`, `semantic_tie_epsilon`.
   - Add backend protocol and adapters in `reranker.py`.
   - Add failing tests for RNK-01/RNK-02/EVAL-03 first.

2. **Wave 1 (pipeline integration)**
   - Integrate semantic reranker in `rank_from_query_embedding()`.
   - Keep lexical/weighted logic available only as tie-breaker path.
   - Enforce deterministic ordering (semantic > weighted > original index).

3. **Wave 2 (evaluation diagnostics)**
   - Extend `main._evaluate()` to compute/print:
     - dense recall@50 (and optionally recall@5 from dense stage),
     - reranked MRR@5,
     - uplift vs lexical baseline,
     - per-language deltas (en/de/fr),
     - reranker latency/throughput.

4. **Wave 3 (ops/perf hardening)**
   - Add model warmup and batch-size controls.
   - Add fallback and explicit error messaging for unavailable backend dependencies.
   - Ensure subset-mode and promotion gate remain deterministic.

## Sources

### Primary (HIGH confidence)
- Phase context and decisions: `.planning/phases/02-semantic-reranking/02-CONTEXT.md`
- Current implementation integration points: `src/clef_retrieval/pipeline.py`, `main.py`, `src/clef_retrieval/retriever.py`, `src/clef_retrieval/config.py`
- Official Jina model card: https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual
- Official BGE model card: https://huggingface.co/BAAI/bge-reranker-v2-m3
- sentence-transformers CrossEncoder docs/source: https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html and https://github.com/UKPLab/sentence-transformers

### Secondary (MEDIUM confidence)
- PyPI registry metadata (versions/dates): torch, transformers, sentence-transformers, FlagEmbedding, einops, huggingface-hub

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: **MEDIUM** — official model cards + PyPI verified; no local execution benchmark yet.
- Architecture: **HIGH** — based on concrete existing code seams and locked phase decisions.
- Pitfalls: **MEDIUM** — grounded in official docs/model constraints, but some runtime risks need local smoke validation.

**Research date:** 2026-03-31  
**Valid until:** 2026-04-07 (fast-moving model/runtime dependencies)
