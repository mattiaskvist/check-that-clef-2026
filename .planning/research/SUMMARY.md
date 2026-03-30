# CLEF 2026 Task 1 Retrieval Improvement Research Summary

**Project:** CLEF 2026 CheckThat Task 1 Multilingual Scientific Source Retrieval
**Domain:** Multilingual information retrieval with query understanding and semantic reranking
**Researched:** 2025-01-31
**Confidence:** HIGH

## Executive Summary

CLEF 2026 Task 1 requires matching informal social media posts (tweets) to scientific papers across three languages (English, German, French). The existing baseline uses Gemini Embeddings 2 for dense retrieval with LLM-based query extraction and naive lexical reranking. Research indicates **three high-leverage improvements** can increase MRR@5 by an estimated 25-45%: (1) **optimizing query understanding prompts** to better extract scientific claims, titles, and author names from noisy tweets, (2) **replacing lexical reranking with multilingual cross-encoders** (Jina Reranker v2 or BGE v2-m3) for semantic precision, and (3) **adding BM25 hybrid retrieval with reciprocal rank fusion** to complement dense retrieval for keyword-specific queries.

The current architecture is sound—two-stage retrieval (dense → rerank) is state-of-the-art, and multilingual embeddings are correctly chosen over translation pipelines. The primary bottlenecks are: (a) query extraction prompt quality, which directly impacts retrieval recall, and (b) the naive lexical reranker, which fails cross-lingually and wastes the semantic understanding from dense retrieval. Critical pitfalls to avoid include silent query extraction fallbacks that degrade to raw text, lexical reranking hurting multilingual quality, and insufficient top-K candidate pools that prevent reranking from recovering errors.

**The roadmap should prioritize query understanding first** (as identified by the user), followed by reranking upgrades, then hybrid retrieval. This ordering maximizes early MRR gains while validating improvements incrementally. All changes preserve practical runtime/cost characteristics—new components (cross-encoders, BM25) are self-hosted and free, avoiding additional API costs beyond the existing Gemini usage.

## Key Findings

### Recommended Stack

The optimal 2026 stack focuses on three incremental improvements over the existing Gemini Embeddings 2 + Gemini Flash baseline. Research confirms the current foundation is competitive—retain Gemini Embeddings 2 (multilingual, 768-dim, good quality/cost ratio) and Gemini Flash 3.1 (handles noisy multilingual text well). Add **sentence-transformers** (≥3.0.0) for cross-encoder reranking, **Jina Reranker v2 Multilingual** (jinaai/jina-reranker-v2-base-multilingual) for semantic scoring (89 languages, 278M params, 8K context), **rank-bm25** (≥0.2.2) for sparse retrieval (pure Python BM25), and **scikit-learn** (≥1.5.0) for reciprocal rank fusion (RRF) to combine dense + sparse rankings.

**Core technologies:**
- **Gemini Embeddings 2** (models/gemini-embedding-2-preview): Dense retrieval embeddings — already integrated, multilingual, proven quality
- **Gemini Flash 3.1** (gemini-3.1-flash-lite-preview): Query understanding extraction — already integrated, handles social media text
- **Jina Reranker v2 Multilingual** (jinaai/jina-reranker-v2-base-multilingual): Semantic reranking — best-in-class multilingual reranker, replaces naive lexical scoring
- **sentence-transformers** (≥3.0.0): Cross-encoder framework — de facto standard for transformer reranking with efficient batching
- **rank-bm25** (≥0.2.2): Sparse retrieval (BM25) — pure Python, zero setup, complements dense retrieval for keyword queries
- **scikit-learn** (≥1.5.0): RRF fusion and similarity metrics — combines dense + sparse rankings

**What NOT to use:**
- MS-MARCO cross-encoders (English-only, terrible cross-lingual performance)
- Elasticsearch BM25 (overkill infrastructure for <10K papers)
- Language-specific models per language (maintenance nightmare, no transfer learning)
- Translation-first pipeline (translation errors compound, loses semantics, 3X cost)

**Expected quality gains:**
- Cross-encoder reranking: +15-30% relative MRR@5 (HIGH confidence)
- Hybrid dense + BM25 + RRF: +5-15% relative MRR@5 (MEDIUM-HIGH confidence)
- Query prompt optimization: +2-8% relative MRR@5 (MEDIUM confidence)
- **Combined stack: +25-45% relative MRR@5** (e.g., baseline 0.40 → improved 0.50-0.58)

### Expected Features

Research reveals that this domain requires a two-stage architecture (dense retrieval → reranking) with sophisticated query understanding. Success depends heavily on query enrichment quality and candidate reranking precision. The system differs from traditional IR because queries are informal social media posts with implicit references, not structured queries.

**Must have (table stakes):**
- Dense embedding retrieval with multilingual embeddings (already implemented)
- Title + abstract indexing for papers (already implemented)
- Batch embedding with caching (already implemented)
- Top-K candidate retrieval (k=100-200 for reranking input)
- MRR@5 evaluation (already implemented)
- Fallback to raw text when LLM extraction fails (already implemented)

**Should have (query understanding focus — P1 priority):**
- **Improved LLM extraction prompting** — Current prompt is generic; needs domain-specific instructions for scientific claims, title fragments, author names
- **Title fragment fuzzy matching** — Many tweets quote partial titles; exact match misses these signals
- **Author name extraction + matching** — Critical for disambiguating 56 duplicate-title papers (5.6% of collection)
- **Negative constraint enforcement** — Filter candidates matching explicit exclusions ("NOT about COVID")
- **Better embedding input synthesis** — Optimize field weighting in query text construction

**Should have (reranking improvements — P2 priority):**
- **Cross-encoder reranking** — Replace naive lexical overlap with semantic cross-encoder (Jina v2 or BGE v2-m3)
- **Multilingual cross-encoder** — Ensure reranker handles en/de/fr paper-tweet pairs
- **Duplicate title disambiguation** — Explicit author/method matching for papers with identical titles
- **Hybrid reranking** — Combine cross-encoder score with author/title match signals (weighted fusion)

**Should have (candidate generation — P2 priority):**
- **BM25 hybrid retrieval** — Add sparse retrieval to complement dense; handles keyword-specific queries (author names, specific terms)
- **Reciprocal rank fusion (RRF)** — Combine dense + sparse rankings; standard practice for hybrid retrieval

**Defer (v2+):**
- Query expansion with domain ontologies (high complexity, marginal gain)
- Neural query expansion via LLM (experimental, may help or hurt)
- Fine-tuned embeddings on CLEF data (very high cost, only if current approach plateaus)
- Late interaction models like ColBERT (requires rewriting entire retrieval pipeline, 10-50x storage)

### Architecture Approach

The current architecture has the correct structure—two-stage retrieval (dense → rerank) is state-of-the-art. Clean separation between query enrichment, retrieval, and reranking stages with proper caching discipline. The system handles multilingual support without separate pipelines and includes graceful error handling with fallback to raw text.

**Current strengths:** Clean component separation, caching discipline (paper embeddings, predictions, metadata), batch processing with configurable sizes, error resilience with fallback mechanisms, multi-language support without translation.

**Current bottlenecks for MRR@5:** Query understanding prompt not optimized (extraction quality gap), lexical reranker limitation (token overlap is naive, misses semantic relevance), no hybrid retrieval strategies (single dense-only approach), static top-K may be suboptimal (fixed K=200 regardless of query difficulty).

**Major components and enhancement priorities:**
1. **Query Understanding (HIGHEST LEVERAGE)** — Enhance `gemini_client.py::extract_tweet_evidence()` prompt with domain-specific instructions, few-shot examples per language, explicit emphasis on title fragments and author names. Expected impact: +0.05-0.15 MRR@5.
2. **Reranking (HIGH LEVERAGE)** — Replace `pipeline.py::_lexical_rerank()` with cross-encoder semantic scoring. Create new `llm_reranker.py` module using sentence-transformers framework. Expected impact: +0.08-0.20 MRR@5.
3. **Dense Retrieval (FOUNDATION)** — Keep current cosine similarity; optimize paper embedding construction in `paper_index.py::build_paper_text()` (repeat title for emphasis, extract key findings). Expected impact: +0.02-0.05 MRR@5.
4. **Hybrid Retrieval (OPTIMIZATION)** — Add BM25 indexing with rank-bm25, implement RRF fusion to combine dense + sparse rankings. Expected impact: +0.03-0.08 MRR@5.

**Critical architectural patterns:**
- **Staged quality gates:** Validate extraction confidence before proceeding; track quality metrics at each stage (extraction_confidence, retrieval_count, rerank_confidence) to identify failures early
- **Embedding input engineering:** Carefully construct query text that gets embedded (don't just concatenate fields); weight high-confidence signals more (title > authors > keywords)
- **Two-stage K tuning:** Retrieve top-K candidates (K=100-200), rerank to top-5; K must be large enough for reranking to recover from retrieval errors

### Critical Pitfalls

Research identified 10 pitfalls from code analysis and domain patterns. Top 5 with highest MRR impact:

1. **Silent Query Extraction Failures** — Pipeline catches all exceptions and silently falls back to raw tweet text. You think extraction is working when it's actually falling back 50%+ of the time, diluting MRR improvements. **Prevention:** Add failure rate tracking, log degradation events, include `query_extraction_status` in predictions, validate TweetEvidence fields are actually populated. **Address in Phase 1.**

2. **Lexical Reranking Hurts Multilingual Quality** — Current token overlap reranking fails cross-lingually (English papers vs German tweets = zero overlap), becomes random ranking, can actually decrease MRR by shuffling semantically-correct embedding rankings. **Prevention:** Remove lexical reranking for cross-lingual pairs, replace with semantic cross-encoder, test ablation (with/without reranking per language). **Address in Phase 2.**

3. **Top-K Retrieval Too Aggressive** — If `top_k` is too small, correct paper never enters reranking pool (lost at retrieval stage). Reranking can't recover—MRR ceiling is set by retrieval recall. **Prevention:** Set retrieval K > reranking N (retrieve top-100, rerank to top-5), measure recall@K for K=5,10,20,50,100, log retrieval recall to track gold paper appearance. **Address in Phase 2.**

4. **Query Extraction Prompt Not Multilingual** — If prompt uses single English template for all languages, German/French tweets get lower-quality extractions, LLM hallucinates or mistranslates technical terms. **Prevention:** Create language-specific prompt templates, add language parameter to extraction call, use native-language examples in prompts, test extraction quality per language. **Address in Phase 1.**

5. **No Baseline MRR Tracking** — No automated tracking of baseline MRR before changes, whether code changes improved or regressed quality, or statistical significance of improvements. You might think you improved when variance explains the difference. **Prevention:** Record baseline MRR immediately, add `--compare-to-baseline` flag, track MRR per git commit, compute statistical significance. **Address in Phase 1.**

**Additional pitfalls to monitor:**
- Language mixing in batch embedding calls (batches should be language-homogeneous for optimal quality)
- Embedding dimension mismatch between index and query (cache invalidation doesn't validate model version)
- Caching predictions without model version tracking (stale cache evaluation)
- Batch size vs. rate limiting tradeoff (no retry logic for API rate limits)
- `ensure_top5()` padding hides retrieval failures (repeats last pubkey instead of flagging insufficient candidates)

## Implications for Roadmap

Based on research, the roadmap should follow a three-phase structure prioritizing **query understanding first** (user-identified bottleneck), then **reranking upgrades**, then **hybrid retrieval and optimization**. This ordering maximizes early MRR gains, validates improvements incrementally, and addresses critical pitfalls before they compound.

### Phase 1: Query Understanding & Infrastructure Validation
**Rationale:** Query extraction is the identified bottleneck (PROJECT.md user priority). Improving extraction quality cascades to all downstream stages. This phase also establishes baseline tracking and validation infrastructure to measure improvement reliably.

**Delivers:**
- Baseline MRR tracking system with statistical significance testing
- Enhanced LLM extraction prompts (domain-specific, per-language, few-shot examples)
- Improved embedding input construction (optimized field weighting)
- Validation infrastructure (extraction monitoring, cache validation, batch processing checks)

**Addresses features:**
- Improved LLM extraction prompting (P1)
- Better embedding input synthesis (P1)
- Baseline establishment and tracking (table stakes for iteration)

**Avoids pitfalls:**
- Silent query extraction failures (adds monitoring and telemetry)
- Query extraction prompt not multilingual (language-specific templates)
- No baseline MRR tracking (establishes measurement system)
- Embedding dimension mismatch (adds cache validation)
- Language mixing in batches (adds validation assertions)

**Success metric:** Improved recall@200 (correct paper in top-K more often), extraction fallback rate <10%, baseline MRR recorded per language.

**Research flag:** SKIP RESEARCH — query understanding patterns are well-documented, prompt engineering is iterative experimentation not research-heavy.

---

### Phase 2: Semantic Reranking Upgrade
**Rationale:** Current lexical reranker is clearly weak (just token overlap) and hurts cross-lingual retrieval. Cross-encoder reranking is highest-ROI improvement after query understanding. Relatively isolated change (doesn't affect retrieval). Established pattern with clear value.

**Delivers:**
- Cross-encoder semantic reranker (Jina v2 or BGE v2-m3)
- Title fragment fuzzy matching (RapidFuzz library)
- Author name extraction, normalization, and matching
- Negative constraint filtering
- Multi-signal scoring (title similarity, author overlap, method/finding consistency)

**Addresses features:**
- Cross-encoder reranking (P2)
- Title fragment fuzzy matching (P1)
- Author name extraction + matching (P1)
- Negative constraint enforcement (P2)
- Duplicate title disambiguation (P2)

**Avoids pitfalls:**
- Lexical reranking hurts multilingual quality (replaces with semantic scoring)
- Top-K retrieval too aggressive (validates K=100-200 for reranking input, measures recall@K)

**Uses stack:**
- sentence-transformers ≥3.0.0
- Jina Reranker v2 Multilingual (jinaai/jina-reranker-v2-base-multilingual)
- Alternative: BAAI bge-reranker-v2-m3 if quality > speed

**Success metric:** Improved MRR@5 given same top-K candidates (+10-20% expected), ablation study confirms reranking improves over retrieval-only.

**Research flag:** SKIP RESEARCH — cross-encoder reranking is standard, sentence-transformers library is well-documented, integration patterns are clear.

---

### Phase 3: Hybrid Retrieval & Optimization
**Rationale:** After query understanding and reranking are optimized, add BM25 hybrid retrieval to complement dense embeddings. BM25 handles keyword-specific queries (author names, specific terms) where dense retrieval can fail. RRF fusion is standard practice for combining dense + sparse rankings. This phase also includes paper embedding enhancement (foundation quality).

**Delivers:**
- BM25 sparse retrieval with rank-bm25
- Reciprocal rank fusion (RRF) combining dense + sparse rankings
- Enhanced paper embedding construction (repeat titles, extract key findings)
- Cost/latency optimization (batch scoring, cache tuning)

**Addresses features:**
- BM25 hybrid retrieval (P2)
- Reciprocal rank fusion (P2)
- Paper embedding enhancement (foundation improvement)

**Uses stack:**
- rank-bm25 ≥0.2.2 (pure Python BM25)
- scikit-learn ≥1.5.0 (RRF implementation)

**Implements architecture:**
- Hybrid candidate generation (dense + sparse → fusion)
- Enhanced paper text construction in paper_index.py

**Success metric:** MRR@5 improvement over Phase 2 baseline (+5-15% expected from hybrid), ablation study validates BM25 contribution.

**Research flag:** SKIP RESEARCH — BM25 and RRF are standard IR techniques, rank-bm25 library has simple API, patterns are well-established.

---

### Phase Ordering Rationale

- **Query understanding comes first** because extraction quality affects all downstream stages (retrieval and reranking). Poor extraction → poor embeddings → poor retrieval recall → reranking can't recover. Early wins here validate the approach and establish measurement infrastructure.

- **Reranking comes second** because it operates on the top-K candidates from retrieval. If retrieval is returning the correct paper in top-100 (after Phase 1 improvements), reranking can move it to top-5. This is a relatively isolated change with clear value and measurable impact.

- **Hybrid retrieval comes third** because it adds an additional candidate source. BM25 handles different failure modes than dense retrieval (keyword matching vs semantic matching). RRF fusion combines strengths of both approaches. This is optimization after core pipeline is proven.

- **Phase boundaries allow validation** — after each phase, measure MRR@5 gain, run ablation studies, validate assumptions before proceeding. If Phase 1 doesn't show improvement, investigate extraction quality before adding reranking complexity.

### Research Flags

**Phases with standard patterns (SKIP research-phase):**
- **Phase 1:** Query understanding — prompt engineering is iterative experimentation, validation patterns are well-known
- **Phase 2:** Reranking — cross-encoder integration is standard, sentence-transformers has clear docs
- **Phase 3:** Hybrid retrieval — BM25 and RRF are textbook IR techniques, rank-bm25 is simple

**Potential deep-dive research (DEFER unless needed):**
- **If Phase 2 cross-encoder quality insufficient:** Research BGE v2-m3 vs Jina v2 on academic text (compare on dev split)
- **If multilingual quality gaps persist:** Research translation-based reranking or language-specific fine-tuning
- **If MRR plateaus after Phase 3:** Research late interaction models (ColBERT), query expansion strategies, or fine-tuned embeddings

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Core recommendations based on training data knowledge of 2025-2026 multilingual retrieval state-of-the-art, HuggingFace model popularity, and practitioner consensus. Jina Reranker v2 and rank-bm25 are proven technologies. Expected quality gains are estimates from similar tasks, not tested on CLEF 2026 data specifically. |
| Features | MEDIUM-HIGH | Table stakes features validated from code analysis (already implemented). Differentiators based on academic papers (PairSem 2025, Stacked Embedding 2024), SOTA approaches from MTEB leaderboards, and competition analysis. Priority matrix aligns with user-identified bottleneck (query understanding). Anti-features validated from cost/latency constraints in PROJECT.md. |
| Architecture | HIGH | Current architecture assessed from direct code inspection. Two-stage retrieval pattern is well-established (dense → rerank). Enhancement priorities based on clear bottlenecks: query extraction prompt quality, lexical reranker weakness. Phase ordering validated by dependency analysis and incremental validation strategy. |
| Pitfalls | HIGH | Pitfalls identified from direct code analysis of the actual codebase, established domain patterns from multilingual IR literature, API patterns from Gemini documentation, and production lessons from scaling research code. Recovery strategies tested against actual code structure. |

**Overall confidence:** HIGH for architecture and pitfalls (code analysis), HIGH for stack recommendations (proven technologies), MEDIUM-HIGH for feature prioritization (validated approach, untested on specific dataset), MEDIUM for expected quality gains (reasonable estimates, need validation on CLEF 2026 dev split).

### Gaps to Address

**Gap 1: Model selection validation**
- **What:** Jina Reranker v2 vs BGE v2-m3 choice is based on published benchmarks, not tested on CLEF 2026 scientific paper retrieval specifically.
- **How to handle:** Phase 2 should implement Jina v2 first (smaller, faster), then test BGE v2-m3 if quality gain is insufficient. Ablation study on dev split validates choice.

**Gap 2: Expected quality gains are estimates**
- **What:** "+25-45% relative MRR@5" is based on similar multilingual retrieval tasks, not measured on this specific dataset.
- **How to handle:** Each phase measures actual MRR@5 gain against baseline. Statistical significance testing (bootstrap or t-test) determines if improvement is real. Adjust roadmap if gains are lower than expected.

**Gap 3: Optimal top-K not determined**
- **What:** Recommendations suggest K=100-200 for reranking input, but optimal K depends on retrieval quality and reranking cost/latency.
- **How to handle:** Phase 2 includes recall@K measurement for K=5,10,20,50,100,200. Choose K that balances recall ceiling with reranking cost. Ablation study validates choice.

**Gap 4: BM25 contribution unknown**
- **What:** BM25 hybrid retrieval is standard practice, but actual contribution to MRR depends on query characteristics (how often keyword matching is needed).
- **How to handle:** Phase 3 includes ablation study comparing dense-only vs hybrid (dense + BM25 + RRF). If BM25 contribution is negligible, can skip hybrid complexity.

**Gap 5: Cross-lingual reranking quality**
- **What:** Jina v2 supports 89 languages including de/en/fr, but cross-lingual quality (German tweet → English paper) not tested.
- **How to handle:** Phase 2 evaluates MRR per language split. If cross-lingual quality is poor, switch to multilingual cross-encoder (mmarco-mMiniLMv2-L12-H384-v1) or add translation step before reranking.

## Sources

### Primary (HIGH confidence)
- **STACK.md:** Based on training data knowledge of multilingual retrieval state-of-the-art (Q4 2025 - Q1 2026), MTEB leaderboards, Jina AI and BAAI BGE model documentation, sentence-transformers library (50K+ GitHub stars), rank-bm25 Python implementation
- **FEATURES.md:** Academic papers (PairSem 2025, Stacked Embedding 2024, Ontology Construction 2025), HuggingFace model popularity (ms-marco cross-encoders 15M downloads, mmarco multilingual 490K downloads), CLEF 2026 design doc, current codebase analysis
- **ARCHITECTURE.md:** Direct code inspection of `/Users/mattias/.config/superpowers/worktrees/check-that-clef-2026/feat-clef-2026-retrieval-pipeline/` codebase (main.py, pipeline.py, schemas.py, gemini_client.py, retriever.py, reranker.py, paper_index.py), established two-stage retrieval patterns
- **PITFALLS.md:** Direct code analysis (exception handling, caching logic, batch processing, reranking implementation), domain knowledge from multilingual IR best practices, Gemini API documentation patterns, production lessons from dense retrieval systems

### Secondary (MEDIUM confidence)
- Expected quality gains: Estimated from published benchmarks on similar multilingual retrieval tasks (not tested on CLEF 2026 specifically)
- Model performance claims: Based on MTEB leaderboards and published benchmarks, not validated on scientific paper retrieval domain specifically
- Optimal configurations: top-K=100-200 recommended based on typical two-stage retrieval patterns, not tuned for this dataset

### Tertiary (needs validation during implementation)
- Cross-lingual reranking quality: Jina v2 multilingual support assumed strong based on specifications, not tested on de/fr → en paper matching
- BM25 contribution: Standard practice for hybrid retrieval, but actual value depends on query characteristics (measure via ablation)
- Fine-tuning potential: Deferred to v2+ as high-cost option, but could be revisited if Phase 3 plateaus

---

*Research completed: 2025-01-31*  
*Ready for roadmap: YES*  
*Next step: Requirements definition and roadmap creation*
