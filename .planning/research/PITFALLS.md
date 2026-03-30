# Pitfalls Research

**Domain:** Multilingual Scientific Source Retrieval with Query Extraction and Reranking
**Researched:** 2025-01-31
**Confidence:** HIGH (based on code analysis and domain patterns)

## Critical Pitfalls

### Pitfall 1: Silent Query Extraction Failures Degrading to Raw Tweet Text

**What goes wrong:**
The pipeline's `build_query_embedding_text()` catches ALL exceptions (APIError, ClientError, ValidationError, TypeError, ValueError, QueryExtractionError) and silently falls back to raw tweet text. This means:
1. LLM extraction failures are invisible in logs
2. You think query extraction is working when it's actually falling back 50%+ of the time
3. MRR improvements from query extraction are diluted by silent fallback cases
4. A/B testing becomes unreliable because you don't know which queries actually used extraction

**Why it happens:**
The code prioritizes robustness over observability. The try-catch in `pipeline.py:68-76` prevents pipeline crashes, but doesn't track failure rates or log degradation events.

**How to avoid:**
1. **Add failure rate tracking**: Count extraction successes vs. fallbacks per batch
2. **Log degradation events**: Warn when fallback happens (at least for first N occurrences)
3. **Add metrics to prediction output**: Include `"query_extraction_status": "success|fallback"` in prediction JSONL
4. **Set quality thresholds**: Alert if fallback rate exceeds 20% for any language
5. **Validate TweetEvidence before accepting**: Check if extracted fields are actually populated, not just validated as schema-compliant

**Warning signs:**
- MRR gains disappear when you increase batch size (more timeouts → more fallbacks)
- German/French splits show no improvement over baseline despite "using" query extraction
- Query extraction model shows low token usage relative to prediction count
- Embedding API usage matches prediction count exactly (means no extraction happened)

**Phase to address:**
Phase 2: Query Extraction Quality Monitoring

---

### Pitfall 2: Language Mixing in Batch Embedding Calls

**What goes wrong:**
The current pipeline processes queries in batches (`query_batch_size`), but batches can contain mixed-language tweets when running on full dev splits. Multilingual embedding models (including Gemini Embeddings 2) perform better when batch contexts are language-homogeneous. Mixed batches create:
1. Suboptimal embedding quality due to language context switching
2. Inconsistent results between single-query and batched evaluation
3. Language-specific MRR variance that appears random but is actually batch-composition dependent

**Why it happens:**
Batching is implemented at the dataset level without language awareness (`main.py:91`). The CLEF dataset splits are monolingual, so this isn't visible yet, but if you combine dev splits or use multilingual evaluation sets, batch heterogeneity degrades quality.

**How to avoid:**
1. **Verify splits are actually monolingual**: Check dataset language tags match expected language
2. **Add language validation**: Assert all tweets in a batch share the same language before embedding
3. **Consider task-level prefixing**: Gemini Embeddings 2 supports task types - use consistent task_type per batch
4. **Document batch homogeneity requirement**: Note in config/docs that batches must be language-homogeneous

**Warning signs:**
- MRR varies significantly with different batch sizes
- Single-tweet evaluation outperforms batch evaluation
- Language pairs show different relative performance when batch size changes
- First batch in each split shows different MRR than subsequent batches

**Phase to address:**
Phase 1: Batch Processing Validation

---

### Pitfall 3: Lexical Reranking Hurts Multilingual Quality

**What goes wrong:**
The current `_lexical_rerank()` in `pipeline.py:36-45` uses simple token overlap between tweet and paper (title + abstract). For multilingual retrieval:
1. **English papers matched against German/French tweets have zero overlap** - lexical reranking becomes random
2. Token-level matching misses semantic similarity across languages
3. Short tokens (length > 2) filter removes many German compound words and French articles
4. Reranking can actually **decrease** MRR by shuffling semantically-correct embedding-based rankings

**Why it happens:**
Lexical reranking is a common information retrieval technique, but it assumes query and document share vocabulary. The code doesn't account for the cross-lingual nature of the CLEF task.

**How to avoid:**
1. **Remove lexical reranking for cross-lingual pairs**: Only apply when tweet language matches paper language
2. **Replace with semantic reranking**: Use cross-encoder model or LLM-based scoring instead
3. **Add language detection**: Detect tweet language and paper language before reranking
4. **Test ablation**: Compare MRR with and without lexical reranking per language split
5. **Consider translation-based reranking**: Translate tweet to English before lexical matching (adds latency/cost)

**Warning signs:**
- German and French splits show lower MRR than English despite same embedding model
- Removing reranking (`rerank_fn=None`) improves MRR
- Papers with English-only titles rank poorly even when semantically relevant
- Top-1 accuracy is worse than top-5 accuracy by large margin (reranking shuffles good candidates down)

**Phase to address:**
Phase 3: Cross-Lingual Reranking Strategy

---

### Pitfall 4: Embedding Dimension Mismatch Between Index and Query

**What goes wrong:**
If embedding model version changes, cache invalidation fails, or configuration drifts, query embeddings and paper index embeddings can have different dimensions. The code uses numpy dot product for similarity (`retriever.py` assumed), which will:
1. Crash with shape mismatch error, OR
2. Silently truncate/pad and return garbage rankings

**Why it happens:**
The caching system (`index_paths()`, `validate_cached_index()`) validates that index rows match metadata count but doesn't store or validate the embedding model version or dimension that created the cache.

**How to avoid:**
1. **Store model version in cache metadata**: Include `embedding_model` and `embedding_dim` in metadata JSONL header
2. **Validate dimensions on load**: Check `embeddings.shape[1]` matches expected dimension for current model
3. **Invalidate cache on model change**: Compare cached model version against config.embedding_model
4. **Add dimension assertion before similarity computation**: Assert query and paper embeddings have matching last dimension
5. **Use semantic versioning for cache**: Name cache files with model version (e.g., `embeddings_gemini-v2_768.npy`)

**Warning signs:**
- Pipeline works after `build-index` but crashes on `predict` with shape errors
- Switching embedding models doesn't trigger automatic index rebuild
- MRR drops to near-zero after config change without index rebuild
- `numpy.dot()` warnings about broadcasting or shape incompatibility

**Phase to address:**
Phase 1: Index Cache Validation

---

### Pitfall 5: Top-K Retrieval Too Aggressive for Reranking

**What goes wrong:**
The current config uses `top_k` for initial retrieval, then lexical reranking within those candidates. If `top_k` is too small:
1. Correct paper never enters reranking pool (lost at retrieval stage)
2. Reranking can't recover - MRR ceiling is set by retrieval recall
3. This is especially problematic for cross-lingual scenarios where embedding similarity is noisier

Looking at the code, `top_k` is used in `rank_from_query_embedding()` (line 102-105) and there's no explicit configuration shown for what `top_k` should be. If it's set to 5, reranking has no room to operate.

**Why it happens:**
Two-stage retrieval often uses K >> N (e.g., retrieve 100, rerank to 5) because fast approximate retrieval is cheap and reranking is expensive. But if both stages use the same K=5, you're just adding latency without improvement potential.

**How to avoid:**
1. **Set retrieval K > reranking N**: Retrieve top-50 or top-100, rerank to top-5
2. **Tune K based on retrieval recall**: Measure recall@K for K=5,10,20,50,100
3. **Balance cost vs. quality**: Larger K increases embedding index scan time but improves reranking ceiling
4. **Make K configurable per stage**: Add `retrieval_k` and `reranking_top_n` as separate config parameters
5. **Log retrieval recall**: Track how often the gold paper appears in top-K before reranking

**Warning signs:**
- Reranking shows no MRR improvement over baseline
- Increasing `top_k` from 5 to 50 shows large MRR jump
- Recall@5 is significantly lower than recall@20
- Ablation study shows retrieval-only performs same as retrieval+reranking

**Phase to address:**
Phase 3: Two-Stage Retrieval Tuning

---

### Pitfall 6: Query Extraction Prompt Not Optimized Per Language

**What goes wrong:**
If `GeminiService.extract_tweet_evidence()` uses a single English prompt for all languages:
1. German and French tweets get lower-quality extractions
2. LLM hallucinates or mistranslates technical terms
3. Multilingual models perform better with language-specific instructions
4. Code-switching in tweets (common in scientific Twitter) confuses single-language prompts

**Why it happens:**
Prompt engineering often starts monolingual and multilingual adaptation is forgotten. The schema (`TweetEvidence`) is language-agnostic, but prompt examples and instructions may not be.

**How to avoid:**
1. **Create language-specific prompt templates**: German prompt for German tweets, etc.
2. **Add language parameter to extraction call**: Pass `lang` through to `extract_tweet_evidence()`
3. **Use native-language examples in prompts**: Show German examples for German extraction
4. **Test extraction quality per language**: Manually review extraction quality for 20 examples per language
5. **Consider language-specific few-shot examples**: Include domain-specific German/French scientific tweet examples

**Warning signs:**
- German and French MRR significantly lower than English despite same pipeline
- Extraction fallback rate higher for non-English languages
- Manual inspection shows English terms in German/French extractions
- LLM token usage inconsistent across languages (suggests fallback differences)

**Phase to address:**
Phase 2: Multilingual Query Extraction

---

### Pitfall 7: Caching Predictions Without Model Version Tracking

**What goes wrong:**
The prediction caching system (`_default_prediction_path`, `_write_predictions`, `_read_predictions`) stores predictions as JSONL but doesn't track:
1. Which embedding model version generated them
2. Whether query extraction was used (`--skip-query-extraction` flag)
3. What `top_k` or batch size was used
4. When the paper index was last rebuilt

This means `--recompute` flag is your only safety mechanism, and you can accidentally evaluate stale predictions after model changes.

**Why it happens:**
Caching improves iteration speed, but metadata tracking is an afterthought. The filename (`predictions_{lang}_{split}.jsonl`) doesn't encode configuration state.

**How to avoid:**
1. **Include model hash in cache filename**: `predictions_{lang}_{split}_{model_hash}.jsonl`
2. **Write config header to prediction files**: First line of JSONL is metadata object with model, date, config
3. **Validate cache freshness on load**: Check cached config matches current config, warn on mismatch
4. **Add `--force` flag**: Explicit override for "I know the cache is stale but use it anyway"
5. **Auto-invalidate on index rebuild**: Touch `.cache/index_version` file, check before loading predictions

**Warning signs:**
- Evaluation results change after cache deletion but not after `--recompute`
- Can't reproduce evaluation scores from previous runs
- Config changes don't affect evaluation until manual cache deletion
- Different team members get different MRR for "same" configuration

**Phase to address:**
Phase 1: Prediction Cache Management

---

### Pitfall 8: Batch Size vs. Rate Limiting Tradeoff Not Documented

**What goes wrong:**
The `query_batch_size` parameter (default from `config.query_batch_size`) controls parallelism, but Gemini API has:
1. QPM (queries per minute) rate limits
2. RPM (requests per minute) rate limits
3. Potential timeout increases with batch size

Larger batches reduce request count but increase per-request timeout risk. Smaller batches increase request count and hit rate limits faster. There's no documentation or auto-tuning for this tradeoff.

**Why it happens:**
The code was likely developed with generous API limits or small test sets. Production use at scale (e.g., full dev splits across 3 languages) exposes rate limiting.

**How to avoid:**
1. **Document optimal batch size range**: Test batch sizes from 1 to 50, document sweet spot
2. **Add retry logic with exponential backoff**: Handle `429 Too Many Requests` errors
3. **Implement adaptive batching**: Reduce batch size after rate limit errors, increase after success
4. **Add `--max-requests-per-minute` flag**: Throttle request rate to stay under limits
5. **Log rate limit encounters**: Track how often you hit limits, tune batch size accordingly

**Warning signs:**
- `APIError` or `ClientError` exceptions increase with larger batch sizes
- Pipeline runs fast initially then stalls for minutes (rate limit recovery)
- Same configuration runs at different speeds on different days (API quota variance)
- Logs show repeated failed requests with retry attempts

**Phase to address:**
Phase 2: API Rate Limit Handling

---

### Pitfall 9: No Baseline MRR Tracking for Regression Detection

**What goes wrong:**
The code can compute MRR@5, but there's no automated tracking of:
1. What the baseline MRR was before changes
2. Whether a code change improved or regressed quality
3. Per-language baseline differences
4. Statistical significance of improvements

You might think you improved MRR when variance explains the difference, or miss regressions because you don't track baseline.

**Why it happens:**
Benchmarking is often manual during research phases. Automation comes later, but by then you've lost baseline history.

**How to avoid:**
1. **Record baseline MRR immediately**: Run baseline evaluation and commit to `.planning/baselines.json`
2. **Add `--compare-to-baseline` flag**: Automatically compare current MRR to recorded baseline
3. **Track MRR per git commit**: Log MRR to `.cache/mrr_history.jsonl` with commit SHA
4. **Compute statistical significance**: Use bootstrap or t-test to determine if improvement is real
5. **Create regression tests**: Fail CI if MRR drops below baseline by >0.01

**Warning signs:**
- Uncertainty about whether changes actually improved quality
- Can't answer "what was MRR before we added query extraction?"
- Lost previous best configuration
- Multiple experiments without clear winner

**Phase to address:**
Phase 1: Baseline Establishment and Tracking

---

### Pitfall 10: `ensure_top5()` Padding Hides Retrieval Failures

**What goes wrong:**
The `ensure_top5()` function (line 20-26 in `pipeline.py`) pads results by repeating the last pubkey if fewer than 5 candidates exist:
```python
while len(values) < 5:
    values.append(values[-1])
```

If retrieval returns 0 candidates (empty results), it returns `["0"] * 5`. This means:
1. Retrieval failures are invisible in MRR evaluation
2. Padding with duplicate pubkeys artificially inflates MRR (if the duplicate happens to be correct, you get credit multiple times)
3. You can't distinguish "only found 2 candidates" from "found 5 candidates"

**Why it happens:**
CLEF evaluation expects exactly 5 predictions per query. Rather than fail, the code pads to meet format requirements.

**How to avoid:**
1. **Log padding events**: Warn when `len(pubkeys) < 5` before padding
2. **Track retrieval depth**: Add `"num_candidates": len(original_candidates)` to prediction output
3. **Use sentinel values for padding**: Pad with `"UNKNOWN_{i}"` instead of repeating last value
4. **Separate MRR computation**: Calculate MRR@K for K=1,2,3,4,5 separately to see padding impact
5. **Investigate root cause of empty results**: If retrieval returns <5 candidates frequently, something is broken

**Warning signs:**
- Many predictions have identical top-5 pubkeys (e.g., `["A", "A", "A", "A", "A"]`)
- MRR@5 > MRR@3 by unexpected margin (suggests padding helped)
- Predictions file shows same pubkey repeated in many top-5 lists
- Collection has >5 papers but predictions often have <5 unique values

**Phase to address:**
Phase 1: Retrieval Completeness Validation

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Catching all extraction exceptions and falling back to raw text | Robust pipeline that never crashes | Silent quality degradation, no observability | Early prototyping only - must add logging before production use |
| Using same prompt for all languages | Faster implementation, single prompt to maintain | Lower quality for non-English, missed optimization opportunity | MVP only - must add language-specific prompts for quality gains |
| Lexical reranking without language detection | Simple implementation, no dependencies | Hurts cross-lingual retrieval, can decrease MRR | Acceptable for monolingual tasks only |
| Flat cache filenames without version hashing | Easy to read, simple file structure | Cache invalidation bugs, stale prediction evaluation | Single-user research phase only |
| No retry logic for API rate limits | Simpler error handling | Pipeline fails on rate limits, requires manual restarts | Small-scale testing only (<100 queries) |

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Gemini Embeddings API | Assuming batch embedding is atomic (all succeed or all fail) | Handle partial batch failures - Gemini returns embeddings for successful items only |
| Gemini LLM extraction | Not setting temperature=0 for deterministic extraction | Use `temperature=0` for reproducible query extraction |
| CLEF Dataset (HuggingFace) | Assuming dataset version is pinned | Pin dataset version in code (e.g., `load_dataset(..., revision="v1.0")`) |
| NumPy embedding storage | Storing embeddings as float64 (default) | Use float32 to halve storage and maintain quality |
| JSONL prediction files | Assuming all JSON parsers handle large numbers identically | Use string pubkeys, not integers, to avoid precision issues |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading entire paper index into memory | Works fine in testing | Use mmap for embeddings, lazy-load metadata | >10K papers or >1GB embeddings |
| Synchronous query extraction for all tweets | Linear time scaling | Batch extraction calls (10-50 tweets per LLM request) | >500 queries in evaluation |
| Recomputing embeddings on every predict | Slow iteration cycle | Cache query embeddings with tweet text hash as key | After first few experiments |
| No progress indicators | Looks stuck, users kill process | Use tqdm for all loops >10 iterations | Any user-facing command |
| Single-threaded batch processing | Underutilizes API quota | Async/parallel batches up to rate limit | >1000 queries or tight deadline |

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Logging full tweet text with PII | GDPR violations if tweets contain names/emails | Hash or truncate logged tweet text, store index only |
| Committing GEMINI_API_KEY to git | API quota theft, unauthorized usage | Use .env file, add to .gitignore, rotate keys regularly |
| Storing raw predictions with identifiable user data | Data breach if cache leaked | Strip PII from cached predictions, anonymize indexes |
| No rate limiting per user/team | One user exhausts shared quota | Implement per-user API budgets or separate keys |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Query extraction:** Implemented but not monitoring fallback rate - verify logs show <10% fallback
- [ ] **Multilingual support:** Processes all languages but prompt is English-only - verify language-specific prompts exist
- [ ] **Caching system:** Saves predictions but no version tracking - verify cache files include model metadata
- [ ] **Batch processing:** Works but no rate limit handling - verify retry logic for 429 errors
- [ ] **MRR evaluation:** Computes score but no baseline comparison - verify baseline recorded and tracked
- [ ] **Reranking:** Implemented but not validated for cross-lingual scenarios - verify ablation study completed
- [ ] **Index validation:** Checks row count but not embedding dimensions - verify dimension match assertion exists
- [ ] **Top-K tuning:** Uses config value but not optimized - verify recall@K measured and K justified
- [ ] **Error handling:** Catches exceptions but doesn't log failure patterns - verify telemetry for extraction/API errors
- [ ] **Reproducibility:** Can rerun but results vary - verify random seeds set and cache keys include all config

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Silent extraction failures | LOW | Add logging, rerun predict with `--recompute`, compare MRR with/without extraction |
| Language mixing in batches | LOW | Add language validation assertion, verify splits are monolingual |
| Lexical reranking hurts quality | LOW | Remove reranking (`rerank_fn=None`), recompute predictions, compare MRR |
| Embedding dimension mismatch | MEDIUM | Rebuild index with `--force-rebuild`, invalidate all prediction caches |
| Top-K too small | LOW | Increase `top_k` to 50-100, rerun predict, measure recall@K improvement |
| Language-specific prompt missing | MEDIUM | Create language-specific prompts, rerun extraction, A/B test per language |
| Stale prediction cache | LOW | Delete cache files, rerun with `--recompute`, add version tracking |
| Rate limit exhaustion | HIGH | Reduce batch size by 50%, add exponential backoff, wait for quota reset (can take hours) |
| No baseline tracking | MEDIUM | Rerun baseline configuration from scratch, record MRR, establish historical tracking |
| Padding hides failures | LOW | Add logging for padding events, investigate why retrieval returns <5 candidates |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Silent query extraction failures | Phase 2: Query Extraction Quality Monitoring | Logs show fallback rate <10%, telemetry dashboard exists |
| Language mixing in batches | Phase 1: Batch Processing Validation | Assertion added, test fails if mixed-language batch created |
| Lexical reranking hurts multilingual quality | Phase 3: Cross-Lingual Reranking Strategy | Ablation study shows reranking improves MRR, or reranking removed |
| Embedding dimension mismatch | Phase 1: Index Cache Validation | Cache metadata includes embedding_dim, validation asserts dimension match |
| Top-K retrieval too aggressive | Phase 3: Two-Stage Retrieval Tuning | Recall@K measured, top_k justified by ablation study |
| Query extraction prompt not multilingual | Phase 2: Multilingual Query Extraction | Language-specific prompts exist, MRR parity across languages |
| Caching predictions without versioning | Phase 1: Prediction Cache Management | Cache filenames include model hash, metadata header validated |
| Batch size vs. rate limiting | Phase 2: API Rate Limit Handling | Retry logic implemented, batch size documented, logs show <1% rate limit errors |
| No baseline MRR tracking | Phase 1: Baseline Establishment and Tracking | Baseline MRR recorded, comparison flag implemented, history logged |
| Padding hides retrieval failures | Phase 1: Retrieval Completeness Validation | Logging added, prediction output includes num_candidates field |

## Sources

- **Code analysis:** `/Users/mattias/.config/superpowers/worktrees/check-that-clef-2026/feat-clef-2026-retrieval-pipeline/` codebase inspection
- **Domain knowledge:** Multilingual information retrieval best practices (CLIR, cross-lingual embeddings, reranking patterns)
- **API patterns:** Gemini API documentation patterns (rate limiting, batch processing, error handling)
- **Retrieval system patterns:** Common pitfalls from dense retrieval, two-stage ranking, neural reranking literature
- **Production lessons:** Typical failures when scaling research code to production benchmarking

---
*Pitfalls research for: CLEF 2026 Task 1 Multilingual Scientific Source Retrieval*
*Researched: 2025-01-31*
*Confidence: HIGH - based on direct code inspection and established domain patterns*
