# Phase 2: Semantic Reranking - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-31
**Phase:** 02-semantic-reranking
**Areas discussed:** Reranker model strategy, Top-K handoff to reranker, Score combination policy, Diagnostics scope for EVAL-03

---

## Reranker model strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Configurable support for both Jina v2 and BGE v2-m3 (default Jina) | Support both backends with runtime selection; default to Jina for initial experiments | ✓ |
| Jina v2 only | Single backend implementation in Phase 2 | |
| BGE v2-m3 only | Single backend implementation in Phase 2 | |

**User's choice:** Configurable support for both Jina v2 and BGE v2-m3 (default Jina)
**Notes:** User selected dual-backend support explicitly; no additional constraints provided.

---

## Top-K handoff to reranker

| Option | Description | Selected |
|--------|-------------|----------|
| 25 | Lower reranker cost, potentially lower recall for rerank stage | |
| 50 | Balanced recall/cost for semantic reranking | ✓ |
| 100 | Higher recall opportunity with higher latency/cost | |
| 200 | Maximum depth, highest cost/latency | |

**User's choice:** 50
**Notes:** Explicitly set as default rerank depth for Phase 2.

---

## Score combination policy

| Option | Description | Selected |
|--------|-------------|----------|
| Semantic-only rerank | Ignore Phase 1 weighted score at rerank stage | |
| Blend semantic + Phase 1 weighted score | Joint weighted mixture for final ordering | |
| Semantic primary, Phase 1 as tie-break | Semantic score drives rank; Phase 1 only resolves ties | ✓ |

**User's choice:** Semantic score primary, Phase 1 weighted score only as tie-break
**Notes:** This locks semantic-first behavior while retaining Phase 1 cues for deterministic tie resolution.

---

## Diagnostics scope for EVAL-03

| Option | Description | Selected |
|--------|-------------|----------|
| Dense retrieval Recall@K | Stage-1 retrieval coverage diagnostic | ✓ |
| Lexical/Phase1 baseline MRR@5 | Explicit lexical baseline reporting in same run | |
| Semantic reranked MRR@5 | Stage-2 semantic performance metric | ✓ |
| Rerank uplift delta (overall + per-language) | Difference between baseline and semantic reranked quality | ✓ |
| Reranker latency/throughput stats | Operational diagnostics for rerank stage | ✓ |
| Candidate truncation stats | Requested K vs reranked K visibility | |

**User's choice:** Dense retrieval Recall@K, Semantic reranked MRR@5, Rerank uplift delta (overall + per-language), Reranker latency/throughput stats
**Notes:** User did not request candidate truncation stats or lexical baseline metric as mandatory output in this discussion pass.

---

## the agent's Discretion

- Tie-break algorithm details (stable ordering mechanics) while preserving semantic-primary policy.
- Exact CLI/config naming for reranker backend selection and rerank depth.
- Diagnostics serialization style if required in addition to human-readable stdout.

## Deferred Ideas

None.
