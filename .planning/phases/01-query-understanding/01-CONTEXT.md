# Phase 1: Query Understanding - Context

**Gathered:** 2026-03-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Improve how tweet signals are extracted and used for retrieval/ranking so query understanding quality measurably improves retrieval outcomes, without adding new capabilities outside Phase 1 scope.

</domain>

<decisions>
## Implementation Decisions

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

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project scope and success targets
- `.planning/PROJECT.md` — Core value, constraints, and experiment-loop policy (subset-first due to prediction latency).
- `.planning/REQUIREMENTS.md` — Phase-1 requirements (`QRY-*`, `EVAL-01/02`, `OPS-*`) and traceability.
- `.planning/ROADMAP.md` §Phase 1 — Goal and success criteria for Query Understanding.
- `.planning/STATE.md` — Current focus and carried decisions relevant to phase planning.

### Existing implementation baseline
- `main.py` — CLI orchestration for predict/evaluate, batching flags, cached-evaluation behavior.
- `src/clef_retrieval/pipeline.py` — Query-text construction, fallback behavior, and ranking composition.
- `src/clef_retrieval/gemini_client.py` — Tweet evidence extraction and embedding client behavior.
- `src/clef_retrieval/config.py` — Current retrieval/query batching config defaults and constraints.
- `scorer.py` — MRR scoring behavior and evaluation assumptions.

### Research guidance
- `.planning/research/SUMMARY.md` — Synthesized direction and phase ordering rationale.
- `.planning/research/FEATURES.md` — Feature prioritization for query understanding and weighting.
- `.planning/research/PITFALLS.md` — Failure modes to avoid in extraction and multilingual evaluation.
- `.planning/research/ARCHITECTURE.md` — Architectural implications for staged quality improvements.
- `.planning/research/STACK.md` — Recommended model/tooling choices and trade-offs.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `build_query_embedding_text()` in `src/clef_retrieval/pipeline.py`: existing extraction+fallback hook for integrating stricter quality gates.
- `TweetEvidence` extraction in `src/clef_retrieval/gemini_client.py`: structured output pathway already in place for field-aware signals.
- `_predict_rows()` in `main.py`: batch predict loop with `--query-batch-size` and `--skip-query-extraction` controls.
- Cached eval path in `_evaluate()` (`main.py`): supports no-recompute evaluation from prediction files.

### Established Patterns
- Pydantic-backed config and schema-driven validation are the dominant pattern in `src/clef_retrieval/`.
- CLI-first experimentation with explicit flags and summary metrics is the established workflow.
- Fail-safe fallback behavior exists for extraction failures; Phase 1 should keep this reliability pattern while adding observability and stricter gating.

### Integration Points
- Extraction quality gating and signal weighting logic should integrate in `src/clef_retrieval/pipeline.py`.
- Per-language experiment reporting and subset policy controls should integrate in `main.py` predict/evaluate flows.
- Optional schema/config extensions should be added in `src/clef_retrieval/schemas.py` and `src/clef_retrieval/config.py`.

</code_context>

<specifics>
## Specific Ideas

- Quality-first default: run structured extraction for all queries in Phase 1.
- Use strict trust criteria for extracted fields; fallback must remain explicit and measurable.
- Keep multilingual fairness explicit by requiring no per-language regression before full-run promotion.
- Lean into reproducible experimentation via seeded fixed subsets to avoid noisy iteration conclusions.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-query-understanding*
*Context gathered: 2026-03-31*
