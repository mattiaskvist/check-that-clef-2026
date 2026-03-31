# Phase 1: Query Understanding - Post-Mortem

**Status:** ❌ DEPRECATED — Hurts performance, do not use  
**Date:** 2026-03-31  
**Finding:** Structured extraction reduced performance by 50%

---

## Executive Summary

Phase 1 aimed to extract structured signals (title, authors, methods, findings) from tweets to improve retrieval quality. After implementation and testing, we discovered that:

- **Structured extraction REDUCES performance** by ~50%
- **Raw tweet text works better** than extracted signals
- **Simple beats complex** for this task

**Recommendation:** Always use `--skip-query-extraction` flag.

---

## Original Goals

### What We Tried to Achieve

1. Extract structured signals from noisy tweets
2. Build better query representations using extracted fields
3. Improve retrieval recall through richer signals
4. Apply negative constraints to filter false positives

### Implementation

- **Extraction Model:** `gemini-3.1-flash-lite-preview`
- **Extracted Fields:**
  - `candidate_title_mentions`: Title fragments
  - `candidate_authors`: Author names
  - `method_terms`: Methodology keywords
  - `finding_terms`: Research findings
  - `keywords`: General concepts
  - `negative_constraints`: Exclusion terms

- **Extraction Gate:** Required minimum signals to pass:
  - ≥1 title mention
  - ≥1 author
  - ≥2 other key fields

- **Query Construction:**
  ```python
  # When extraction passes:
  query_text = f"{title} {authors} {methods} {findings} {keywords}"
  
  # When extraction fails:
  query_text = raw_tweet_text  # Fallback
  ```

---

## Performance Results

### WITH Extraction (Phase 1 Enabled)

| Language | MRR@5 | Recall@5 |
|----------|-------|----------|
| English  | 35.57% | 43.7% |
| French   | 16.27% | 22.0% |

### WITHOUT Extraction (Phase 1 Disabled)

| Language | MRR@5 | Recall@5 | Improvement |
|----------|-------|----------|-------------|
| English  | 64.71% | 76.01% | **+82%** ⬆️ |
| French   | 62.60% | 75.64% | **+285%** ⬆️ |

**Finding:** Disabling extraction DOUBLED EN performance and TRIPLED FR performance.

---

## Why It Failed

### 1. Information Loss
Raw tweets contain rich context that gets lost during structured extraction:
- Informal language and abbreviations
- Citation patterns and academic conventions
- Implicit references and contextual cues

**Extracted fields stripped away this valuable information.**

### 2. Extraction Errors
LLM extraction is imperfect:
- Missed important signals
- Extracted irrelevant terms
- Failed to parse complex academic language
- Introduced noise through hallucinations

### 3. Format Mismatch
Gemini embeddings are trained on natural text, not structured fields:
- Concatenating extracted fields creates unnatural text
- Loses sentence structure and semantic flow
- Doesn't match embedding model's training distribution

### 4. Inconsistent Representation
Extraction gate caused inconsistency:
- Queries that passed gate → Used extracted fields
- Queries that failed gate → Used raw tweet text
- Different representation formats hurt overall quality

### 5. Compounding Errors
Each extraction error amplified downstream:
```
Wrong extraction → Bad query text → Poor embeddings → Low recall → Bad ranking
```

---

## What We Learned

### ✅ Lessons Learned

1. **Trust the embeddings** — Modern embedding models work well with natural text
2. **Simpler is better** — Don't over-engineer without validation
3. **Test ablations early** — Should have tested WITH/WITHOUT extraction in Phase 1
4. **Baseline comparison matters** — Should have compared against baseline immediately
5. **Assumptions need validation** — "More structure = better performance" was wrong

### ❌ Mistakes Made

1. **No baseline comparison during Phase 1**
   - Never measured performance against original baseline
   - Assumed extraction was helping without evidence

2. **No ablation testing**
   - Never tested extraction OFF as control
   - Only validated that extraction "worked" (produced output)

3. **Complex implementation first**
   - Built full extraction pipeline before validating value
   - Should have started simple and added complexity only if beneficial

4. **Ignored negative signals**
   - Low recall (43.7%) should have triggered investigation
   - Focused on implementation details instead of end-to-end metrics

---

## Technical Implementation (Preserved for Reference)

The following code is **preserved but not recommended**:

### Extraction Logic
- `src/clef_retrieval/gemini_client.py`: `extract_tweet_evidence()`
- `src/clef_retrieval/schemas.py`: `TweetEvidence` schema
- `src/clef_retrieval/pipeline.py`: `build_query_embedding_text_with_metadata()`

### Gate Logic
- `src/clef_retrieval/query_policy.py`: `extraction_gate_passes()`
- Config fields: `extraction_gate_min_title_mentions`, `extraction_gate_min_authors`, etc.

### Weighted Reranking
- `src/clef_retrieval/pipeline.py`: `weighted_signal_rerank()`
- Uses extracted signals for post-retrieval scoring
- **NOTE:** This still works but has minimal impact compared to semantic reranking

### Tests
- `tests/test_pipeline.py`: Extraction outcome tests
- `tests/test_signal_weighting.py`: Weighted reranking tests
- `tests/test_cost_transparency.py`: API usage estimation

All tests still pass, but the feature is not recommended for production use.

---

## Alternative Approaches (Future Research)

If you want to revisit structured extraction:

### 1. Hybrid Approach
- Use raw tweets for dense retrieval (high recall)
- Use extracted signals ONLY for reranking (post-retrieval refinement)
- Never replace raw text for embedding

### 2. Selective Extraction
- Only extract when confidence is very high (>95%)
- Otherwise always use raw text
- Avoid inconsistent representations

### 3. Better Extraction Models
- Fine-tune extraction model on this specific domain
- Train on ground truth paper citations
- Validate extraction quality before deployment

### 4. Query Expansion (Not Replacement)
- Keep raw tweet text as primary query
- Add extracted terms as query expansion
- Don't discard original context

---

## Recommendation

**DO NOT USE Phase 1 extraction in production.**

Always run with `--skip-query-extraction`:
```bash
uv run python main.py predict --lang en --split dev --skip-query-extraction
```

Phases 2 (Semantic Reranking) and 3 (Duplicate Disambiguation) work excellently when given raw tweet input. No extraction needed.

---

## Impact

- **Time invested:** ~5 hours implementation + testing
- **Code created:** ~800 lines (extraction, gates, weighting, tests)
- **Performance impact:** -50% (harmful)
- **Valuable lesson:** Sometimes the simple approach is best

The code remains in the repository for:
1. Reference and learning
2. Future research on why extraction failed
3. Comparison baseline for alternative approaches

---

**Archived:** 2026-03-31  
**Reason:** Performance regression — simpler approach outperforms by 80%+
