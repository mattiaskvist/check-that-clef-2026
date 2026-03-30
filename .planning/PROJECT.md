# CLEF 2026 CheckThat Task 1 Retrieval Improvement

## What This Is

This project focuses on improving a multilingual source-retrieval pipeline for CLEF 2026 CheckThat Task 1. Given a social media post with an implicit scientific paper reference, the system should retrieve the correct paper from a candidate pool. The immediate goal is to improve quality over the current in-repo baseline while keeping iteration speed practical.

## Core Value

Improve MRR@5 on the dev split over the repository's current baseline for Task 1 source retrieval.

## Requirements

### Validated

- ✓ Load and evaluate Task 1 data across English, German, and French splits — existing
- ✓ Build paper embeddings and run top-k similarity retrieval against collection candidates — existing
- ✓ Run prediction and scoring workflows from CLI with MRR@5 evaluation — existing

### Active

- [ ] Improve query understanding (tweet extraction and prompting) to increase retrieval quality
- [ ] Integrate higher-signal query representations into embedding retrieval
- [ ] Compare quality improvements against the current in-repo baseline on dev MRR@5
- [ ] Preserve practical runtime/cost characteristics while improving quality

### Out of Scope

- Task 2 or other CLEF tasks — outside this milestone's objective
- Building a web UI or product frontend — not required for retrieval benchmarking
- Large infrastructure/deployment work — unnecessary for current experimentation loop
- Collecting new external datasets — benchmark should remain comparable to current setup

## Context

The repository already contains a working retrieval pipeline with dataset loaders, embedding/indexing flow, prediction, and evaluation utilities. Recent work introduced batch embedding controls, progress reporting, and cached prediction reuse during evaluation. The user has chosen Gemini Embeddings 2 and LLM-assisted tweet extraction as the core direction and is currently blocked on beating the existing baseline.

The workflow is experimentation-heavy and metric-driven. The user priority is quality gain first (MRR@5 uplift), with speed as a secondary preference and a currently moderate API spend cap that can be increased if justified.

## Constraints

- **Task Scope**: CLEF 2026 CheckThat Task 1 only — project focus is constrained to source retrieval for scientific web claims
- **Model Direction**: Use Gemini Embeddings 2 plus LLM-assisted query understanding — chosen approach should stay consistent
- **Evaluation Target**: Dev MRR@5 vs in-repo baseline — progress must be measured against this metric
- **Data Boundary**: Existing CLEF dataset splits only — no new external dataset ingestion for this initiative
- **Cost Awareness**: Initial spend cap around 200 SEK — improvements should be mindful of API usage and query-time overhead

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Prioritize query-understanding improvements first | User identified extraction/prompting as highest-leverage bottleneck | — Pending |
| Measure success against current in-repo baseline, not notebook baseline | Matches user's explicit success definition for this iteration | — Pending |
| Keep the project as a retrieval research workflow (no UI/deployment expansion) | Prevents scope drift and keeps focus on leaderboard-relevant quality gains | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-31 after initialization*
