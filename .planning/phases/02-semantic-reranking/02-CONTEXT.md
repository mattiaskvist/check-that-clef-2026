# Phase 2: Semantic Reranking - Context

**Gathered:** 2026-03-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace Phase 1 lexical reranking with multilingual semantic reranking over dense-retrieval candidates, while preserving Phase 1 stability constraints (subset-first experimentation, no per-language regression tolerance, and operational transparency).

</domain>

<decisions>
## Implementation Decisions

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

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope and acceptance targets
- `.planning/ROADMAP.md` §Phase 2: Semantic Reranking — phase goal, dependencies, and success criteria.
- `.planning/REQUIREMENTS.md` — `RNK-01`, `RNK-02`, `EVAL-03` requirements and traceability table.
- `.planning/PROJECT.md` — milestone constraints (cost-awareness, subset-first experiment loop, benchmark comparability).
- `.planning/STATE.md` — current execution position and continuity notes.

### Phase 1 behavior that must remain compatible
- `.planning/phases/01-query-understanding/01-CONTEXT.md` — locked extraction/normalization/subset policy decisions.
- `.planning/phases/01-query-understanding/01-VERIFICATION.md` — verified behaviors that Phase 2 must not regress.
- `src/clef_retrieval/pipeline.py` — current retrieval/rerank flow and Phase 1 weighted rerank behavior.
- `src/clef_retrieval/query_policy.py` — seeded subset and promotion/no-regression helpers carried forward.
- `main.py` — current CLI telemetry, multilingual reporting, and cost guardrail behavior.

### Existing implementation assets for rerank integration
- `src/clef_retrieval/retriever.py` — dense candidate retrieval and top-K selection primitives.
- `src/clef_retrieval/reranker.py` — reranker result schema and top-5 clipping helper.
- `src/clef_retrieval/config.py` — retrieval configuration model and validation conventions.
- `tests/test_signal_weighting.py` — current ordering/constraint expectations used as baseline comparison.
- `tests/test_language_metrics_reporting.py` and `tests/test_cost_transparency.py` — evaluation/ops diagnostics patterns to extend.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `retrieve_top_pubkeys()` in `src/clef_retrieval/retriever.py` provides deterministic dense top-K handoff for semantic rerank input.
- `RerankResult` in `src/clef_retrieval/reranker.py` provides an existing typed structure for semantic rerank outputs.
- `should_promote_from_subset()` in `src/clef_retrieval/query_policy.py` already enforces no-language-regression promotion semantics.
- `_estimate_query_api_usage()` and cost guardrail helpers in `main.py` can be extended to include reranker-stage usage/latency accounting.

### Established Patterns
- Config-first behavior control via `RetrievalConfig` with Pydantic validation.
- Deterministic, test-first policy evolution with focused unit/CLI tests.
- Evaluation output currently emphasizes transparent operational summaries; Phase 2 diagnostics should follow the same style.

### Integration Points
- `rank_from_query_embedding()` in `src/clef_retrieval/pipeline.py` is the primary insertion point for semantic reranker invocation after dense retrieval.
- `main.py::_evaluate` is the primary insertion point for stage-level diagnostics and uplift reporting (`EVAL-03`).
- `tests/test_pipeline.py`, `tests/test_cli.py`, and new Phase 2 tests should lock semantic-vs-lexical behavior and multilingual non-regression signals.

</code_context>

<specifics>
## Specific Ideas

- Keep semantic reranking backend-swappable from day one (Jina default, BGE available) to avoid rework when comparing multilingual behavior.
- Favor deterministic tie-break semantics so repeated seeded subset runs remain reproducible.
- Make diagnostics actionable: each run should show whether bottleneck is candidate retrieval quality or reranker ordering quality.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-semantic-reranking*
*Context gathered: 2026-03-31*
