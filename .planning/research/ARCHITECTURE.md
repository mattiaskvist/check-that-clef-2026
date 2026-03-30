# Architecture Research: CLEF Task 1 Retrieval Pipeline

**Domain:** Scientific source retrieval for social media claims
**Researched:** 2025-01-21
**Confidence:** HIGH

## Current Architecture Assessment

### Existing System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       CLI Entry (main.py)                        │
│         [build-index] [predict] [evaluate]                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                      Pipeline Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  build_query_embedding_text() → rank_from_query_embedding()     │
│  predict_top5_with_embeddings()                                  │
└────────┬────────────────────┬─────────────────┬─────────────────┘
         │                    │                 │
    ┌────┴─────┐      ┌──────┴──────┐    ┌────┴─────────┐
    │ Gemini   │      │  Retriever  │    │  Reranker    │
    │ Service  │      │  (cosine)   │    │  (lexical)   │
    └────┬─────┘      └──────┬──────┘    └────┬─────────┘
         │                   │                 │
┌────────┴───────────────────┴─────────────────┴─────────────────┐
│                        Data Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Paper Index  │  │  Query Data  │  │ Predictions  │          │
│  │  (.npy cache)│  │ (HF dataset) │  │   (.jsonl)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Current Component Responsibilities

| Component | Current Responsibility | Quality Impact |
|-----------|------------------------|----------------|
| `gemini_client.py` | LLM query extraction + embeddings | **Critical** - extraction quality directly affects MRR@5 |
| `retriever.py` | Dense cosine similarity search | **High** - recall bottleneck (must include correct paper in top-K) |
| `reranker.py` | Lexical token overlap scoring | **Medium** - simple fallback, limited semantic understanding |
| `pipeline.py` | Orchestration + error handling | **Support** - holds quality stages together |
| `paper_index.py` | Offline paper embedding cache | **Foundation** - quality depends on paper representation |

### Current Data Flow

```
Tweet Text
    ↓
[extract_tweet_evidence] → TweetEvidence schema
    ↓
[build_embedding_input] → query_text_for_embedding
    ↓
[embed_texts] → query_embedding (768-dim)
    ↓
[retrieve_top_pubkeys] → top-K candidates (K=200)
    ↓
[_lexical_rerank] → top-5 final predictions
    ↓
MRR@5 Evaluation
```

### Architecture Strengths

1. **Clean separation**: Query enrichment → Retrieval → Reranking stages are decoupled
2. **Caching discipline**: Paper embeddings, predictions, and metadata persist across runs
3. **Batch processing**: Efficient API usage with configurable batch sizes
4. **Error resilience**: Graceful fallback to raw text when extraction fails
5. **Multi-language support**: Schema handles en/de/fr without separate pipelines

### Architecture Weaknesses (MRR@5 Bottlenecks)

1. **Query understanding gap**: Current `TweetEvidence` schema exists but extraction prompt/logic not optimized
2. **Lexical reranker limitation**: Token overlap is naive; misses semantic relevance and authority signals
3. **No candidate expansion**: Single embedding-only retrieval; no hybrid/fusion strategies
4. **Missing feedback loop**: No iterative refinement or query rewriting
5. **Static top-K**: Fixed K=200 may be too conservative or wasteful depending on query difficulty

## Recommended Architecture for MRR@5 Uplift

### Target System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  Enhanced Pipeline Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Stage 0: Query Understanding                             │   │
│  │  • Enhanced TweetEvidence extraction                     │   │
│  │  • Multi-facet query construction                        │   │
│  │  • Confidence scoring for each extracted field           │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Stage 1: Candidate Generation (Retrieval)                │   │
│  │  • Dense retrieval (current: cosine similarity)          │   │
│  │  • Optional: Hybrid with BM25/lexical fallback           │   │
│  │  • Adaptive top-K selection                              │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Stage 2: Reranking                                       │   │
│  │  • LLM-based semantic reranker (replaces lexical)        │   │
│  │  • Multi-signal scoring: title match, author overlap,    │   │
│  │    method/finding consistency, domain alignment          │   │
│  │  • Confidence-aware final ranking                        │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ↓                                      │
│                      Top-5 Predictions                           │
└─────────────────────────────────────────────────────────────────┘
```

### Component Enhancement Boundaries

| Component | Enhancement | Implementation Complexity | Expected MRR@5 Impact |
|-----------|-------------|---------------------------|----------------------|
| **Query Understanding** | Optimized extraction prompt + field validation | Low-Medium | **High** (0.05-0.15 gain) |
| **Dense Retrieval** | Keep current cosine; optimize paper embeddings | Low | Medium (0.02-0.05 gain) |
| **Reranking** | Replace lexical with LLM multi-signal reranker | Medium | **High** (0.08-0.20 gain) |
| **Candidate Generation** | Adaptive top-K or hybrid retrieval | Medium-High | Medium (0.03-0.08 gain) |

### Recommended Build Order (Phase Structure)

#### Phase 1: Query Understanding Uplift (Highest Leverage)
**Goal:** Improve extraction quality to get better query embeddings
**Confidence:** HIGH - direct impact, low risk, fast iteration

**Why first:**
- Current `extract_tweet_evidence()` returns structured schema but extraction quality unknown
- Gemini Flash-Lite with better prompting is fast and cheap
- Improvements here cascade to all downstream stages
- Easy to evaluate in isolation (extract → embed → retrieve)

**Components to modify:**
- `gemini_client.py::extract_tweet_evidence()` - enhance prompt with examples
- `schemas.py::TweetEvidence` - potentially add confidence scores per field
- `gemini_client.py::build_embedding_input()` - optimize field weighting for embedding

**Success metric:** Improved recall@200 (correct paper in top-K more often)

**Architecture pattern:**
```python
# Enhanced extraction with confidence
class TweetEvidence(BaseModel):
    claim_summary: str
    claim_confidence: float = 0.0  # How confident is this a scientific claim?
    candidate_title_mentions: list[str]
    title_confidence: float = 0.0  # How explicit are title hints?
    candidate_authors: list[str]
    author_confidence: float = 0.0
    # ... existing fields ...
    
# Weighted embedding input construction
def build_embedding_input(evidence: TweetEvidence, raw_tweet: str) -> str:
    """Construct embedding input prioritizing high-confidence fields."""
    parts = []
    
    # Core claim (always include)
    if evidence.claim_summary:
        parts.append(f"Claim: {evidence.claim_summary}")
    
    # High-signal fields (weight by confidence)
    if evidence.candidate_title_mentions and evidence.title_confidence > 0.5:
        parts.append(f"Title hints: {', '.join(evidence.candidate_title_mentions)}")
    
    if evidence.candidate_authors and evidence.author_confidence > 0.5:
        parts.append(f"Authors: {', '.join(evidence.candidate_authors)}")
    
    # Domain/method context
    if evidence.keywords:
        parts.append(f"Keywords: {', '.join(evidence.keywords[:5])}")
    
    # Fallback to raw if extraction weak
    if not parts or evidence.claim_confidence < 0.3:
        return raw_tweet
    
    return " | ".join(parts)
```

**Pitfalls to avoid:**
- Over-engineering extraction schema - keep it simple, focus on what helps retrieval
- Not measuring extraction quality - log extraction failures and partial extractions
- Prompt drift - version control prompt templates, track what works

#### Phase 2: Reranking Upgrade (Second Highest Leverage)
**Goal:** Replace naive lexical reranker with LLM-based semantic scoring
**Confidence:** HIGH - established pattern, clear value

**Why second:**
- Current lexical reranker is clearly weak (just token overlap)
- Top-K from retrieval likely contains correct answer but ranked poorly
- LLM reranker can reason about multiple signals simultaneously
- Relatively isolated change (doesn't affect retrieval)

**Components to modify:**
- New module: `clef_retrieval/llm_reranker.py`
- `pipeline.py::stage2_rerank()` - integrate new reranker
- `schemas.py` - add `RerankSignals` schema for structured scoring

**Architecture pattern:**
```python
class RerankSignals(BaseModel):
    """Multi-signal relevance evidence."""
    title_similarity: float  # 0-1, exact/fuzzy title match
    author_overlap: float    # 0-1, fraction of authors matched
    method_consistency: float  # 0-1, method terms alignment
    finding_consistency: float  # 0-1, findings/results alignment
    domain_alignment: float   # 0-1, domain terms match
    venue_year_hints: float   # 0-1, temporal/venue context
    overall_score: float      # 0-1, weighted combination
    rationale: str            # Brief explanation

class LLMReranker:
    """Gemini-based multi-signal reranker."""
    
    def rerank_candidates(
        self,
        tweet_evidence: TweetEvidence,
        candidates: list[PaperEvidence],
    ) -> list[tuple[str, RerankSignals]]:
        """
        Score each candidate against tweet evidence.
        Returns: [(pubkey, signals), ...] sorted by overall_score desc
        """
        prompt = self._build_rerank_prompt(tweet_evidence, candidates)
        response = self.service.generate_structured(
            prompt=prompt,
            schema=list[RerankSignals],
            model=self.config.rerank_model,
        )
        return self._sort_by_signals(candidates, response)
    
    def _build_rerank_prompt(self, tweet: TweetEvidence, papers: list[PaperEvidence]) -> str:
        """
        Build prompt that asks LLM to score each paper against tweet signals.
        Include few-shot examples of good vs bad matches.
        """
        return f"""
        You are a scientific paper retrieval expert. Given a social media post 
        about a paper and {len(papers)} candidate papers, score each candidate's 
        relevance using multiple signals.
        
        Tweet evidence:
        - Claim: {tweet.claim_summary}
        - Title hints: {tweet.candidate_title_mentions}
        - Authors: {tweet.candidate_authors}
        - Keywords: {tweet.keywords}
        - Method terms: {tweet.method_terms}
        - Finding terms: {tweet.finding_terms}
        
        Candidates:
        {self._format_candidates(papers)}
        
        For each candidate, output a RerankSignals object with:
        - title_similarity: how well does title match hints? (0-1)
        - author_overlap: fraction of mentioned authors present (0-1)
        - method_consistency: do methods align? (0-1)
        - finding_consistency: do results/findings align? (0-1)
        - domain_alignment: same field/domain? (0-1)
        - venue_year_hints: temporal/venue context match? (0-1)
        - overall_score: weighted combination emphasizing title+authors (0-1)
        - rationale: 1-2 sentence explanation
        
        Return list of {len(papers)} RerankSignals objects, one per candidate.
        """
```

**Success metric:** Improved MRR@5 given same top-K candidates

**Pitfalls to avoid:**
- Sending all K=200 candidates to reranker - too slow/expensive, rerank top-20 or top-50
- Not validating signal weights - some signals matter more (title >> keywords)
- Ignoring multi-language - reranker must handle en/de/fr paper-tweet pairs

#### Phase 3: Paper Embedding Enhancement (Foundation Quality)
**Goal:** Improve paper representations in the index
**Confidence:** MEDIUM-HIGH - indirect impact, requires re-indexing

**Why third:**
- Retrieval recall depends on paper embedding quality
- Current implementation uses simple title+authors+abstract
- Enhancing paper side is cheaper than query side (offline, one-time cost)
- Must re-build index, so defer until after query/rerank proven

**Components to modify:**
- `paper_index.py::build_paper_text()` - enhance paper representation
- `paper_index.py::_build_metadata_row()` - extract richer metadata
- Optional: Add LLM-based paper enrichment (keywords, method/finding terms)

**Architecture pattern:**
```python
def build_enhanced_paper_text(evidence: PaperEvidence) -> str:
    """
    Construct paper text optimized for retrieval matching against tweets.
    Priority: title > authors > key findings > methods > abstract.
    """
    parts = []
    
    # Title (repeat for emphasis - tweets often mention title fragments)
    parts.append(f"Title: {evidence.title}")
    parts.append(f"Paper title: {evidence.title}")
    
    # Authors (critical for disambiguation)
    if evidence.authors:
        parts.append(f"Authors: {evidence.authors}")
    
    # Abstract highlights (first + last sentence often contain key findings)
    if evidence.abstract:
        sentences = evidence.abstract.split('. ')
        if len(sentences) > 0:
            parts.append(f"Main finding: {sentences[0]}")
        if len(sentences) > 2:
            parts.append(f"Conclusion: {sentences[-1]}")
    
    # Extracted terms (if using LLM enrichment)
    if evidence.method_terms:
        parts.append(f"Methods: {', '.join(evidence.method_terms[:5])}")
    
    if evidence.finding_terms:
        parts.append(f"Findings: {', '.join(evidence.finding_terms[:5])}")
    
    # Full abstract last (for general semantic context)
    if evidence.abstract:
        parts.append(f"Abstract: {evidence.abstract}")
    
    return " | ".join(parts)
```

**Success metric:** Improved recall@200 with same query quality

**Pitfalls to avoid:**
- Over-complicating paper enrichment - start simple, measure impact
- Not A/B testing representations - compare old vs new index on dev set
- Ignoring index rebuild cost - 10K papers * embedding cost adds up

#### Phase 4: Adaptive Candidate Generation (Optional Optimization)
**Goal:** Dynamically adjust top-K based on query confidence
**Confidence:** MEDIUM - optimization, diminishing returns

**Why fourth (optional):**
- Fixed K=200 works but wastes compute on easy queries, undershoots on hard queries
- Requires query difficulty estimation
- More complex, less clear MRR@5 gain
- Only pursue if Phases 1-3 don't hit target

**Components to modify:**
- `pipeline.py::rank_from_query_embedding()` - adaptive K selection
- `config.py` - add K range parameters (min_k, max_k, default_k)

**Architecture pattern:**
```python
def estimate_query_difficulty(tweet_evidence: TweetEvidence) -> float:
    """
    Estimate retrieval difficulty based on evidence strength.
    Returns: difficulty score 0-1 (0=easy, 1=hard)
    """
    difficulty = 0.5  # baseline
    
    # Easy signals (reduce difficulty)
    if tweet_evidence.title_confidence > 0.7:
        difficulty -= 0.3
    if tweet_evidence.author_confidence > 0.6:
        difficulty -= 0.2
    
    # Hard signals (increase difficulty)
    if not tweet_evidence.candidate_title_mentions:
        difficulty += 0.2
    if tweet_evidence.claim_confidence < 0.4:
        difficulty += 0.1
    
    return max(0.0, min(1.0, difficulty))

def adaptive_top_k(base_k: int, difficulty: float, min_k: int = 50, max_k: int = 400) -> int:
    """Scale K based on estimated query difficulty."""
    k = int(base_k + (max_k - base_k) * difficulty)
    return max(min_k, min(k, max_k))
```

**Success metric:** Same MRR@5 with lower average K (cost/latency win)

**Pitfalls to avoid:**
- Premature optimization - only do this if Phases 1-3 succeed but need efficiency
- Complexity explosion - keep difficulty estimation simple
- Over-tuning on dev set - validate on held-out data

## Critical Architectural Patterns

### Pattern 1: Staged Quality Gates

**What:** Each pipeline stage validates output quality before proceeding
**When to use:** Multi-stage pipelines where bad input cascades downstream
**Trade-offs:** Adds complexity but prevents silent failures

**Example:**
```python
def predict_top5_with_quality_gates(
    tweet_text: str,
    service: GeminiService,
    paper_embeddings: np.ndarray,
    metadata_rows: list[dict],
    top_k: int,
) -> tuple[list[str], dict[str, float]]:
    """Predict top-5 with quality metrics at each stage."""
    
    quality_metrics = {}
    
    # Stage 0: Query understanding
    evidence = service.extract_tweet_evidence(tweet_text)
    quality_metrics['extraction_confidence'] = evidence.claim_confidence
    
    if evidence.claim_confidence < 0.1:
        # Extraction failed, use raw text fallback
        query_text = tweet_text
        quality_metrics['extraction_status'] = 'fallback'
    else:
        query_text = build_embedding_input(evidence, tweet_text)
        quality_metrics['extraction_status'] = 'success'
    
    # Stage 1: Retrieval
    query_embedding = service.embed_texts([query_text])[0]
    candidates = retrieve_top_pubkeys(
        query_embedding, paper_embeddings, 
        [r['pubkey'] for r in metadata_rows], k=top_k
    )
    quality_metrics['retrieval_count'] = len(candidates)
    
    # Stage 2: Reranking
    if len(candidates) < 5:
        # Insufficient candidates, pad with fallback
        quality_metrics['rerank_status'] = 'insufficient_candidates'
        return ensure_top5(candidates), quality_metrics
    
    reranked = rerank_candidates(tweet_text, candidates, metadata_rows)
    quality_metrics['rerank_status'] = 'success'
    quality_metrics['rerank_confidence'] = reranked[0].overall_score if reranked else 0.0
    
    return ensure_top5([r.pubkey for r in reranked]), quality_metrics
```

**Why this matters for MRR@5:**
- Identifies which stage is failing (low extraction confidence? low rerank confidence?)
- Enables targeted debugging and prompt improvement
- Supports ablation studies (turn off stages to isolate impact)

### Pattern 2: Embedding Input Engineering

**What:** Carefully construct the text that gets embedded, don't just concatenate fields
**When to use:** Dense retrieval with semantic embeddings
**Trade-offs:** Must balance information density vs noise

**Example:**
```python
def build_retrieval_optimized_input(
    evidence: TweetEvidence,
    raw_tweet: str,
    mode: str = "balanced"
) -> str:
    """
    Construct embedding input optimized for paper retrieval.
    
    Modes:
    - "title_focused": Emphasize title/author signals
    - "semantic": Emphasize claim/findings
    - "balanced": Mix of both
    """
    
    if mode == "title_focused":
        # For queries with strong title/author hints
        parts = []
        if evidence.candidate_title_mentions:
            parts.append(" ".join(evidence.candidate_title_mentions))
        if evidence.candidate_authors:
            parts.append(" ".join(evidence.candidate_authors))
        if evidence.claim_summary:
            parts.append(evidence.claim_summary)
        return " | ".join(parts) if parts else raw_tweet
    
    elif mode == "semantic":
        # For vague queries, rely on semantic matching
        parts = [evidence.claim_summary] if evidence.claim_summary else []
        parts.extend(evidence.finding_terms[:3])
        parts.extend(evidence.method_terms[:3])
        parts.extend(evidence.domain_terms[:2])
        return " ".join(parts) if parts else raw_tweet
    
    else:  # balanced (recommended default)
        # Weight high-confidence signals more
        parts = []
        
        # Always include claim
        if evidence.claim_summary:
            parts.append(evidence.claim_summary)
        
        # Title hints (very high signal)
        if evidence.candidate_title_mentions:
            parts.append("Title: " + " ".join(evidence.candidate_title_mentions))
        
        # Authors (high signal)
        if evidence.candidate_authors:
            parts.append("Authors: " + " ".join(evidence.candidate_authors))
        
        # Domain/method context (medium signal)
        context = []
        context.extend(evidence.method_terms[:3])
        context.extend(evidence.finding_terms[:3])
        if context:
            parts.append("Context: " + " ".join(context))
        
        return " | ".join(parts) if parts else raw_tweet
```

**Why this matters for MRR@5:**
- Embedding models are sensitive to input structure
- Different query types need different representations
- Avoids diluting high-signal terms with noise

### Pattern 3: Multi-Signal Reranking

**What:** Combine multiple relevance signals rather than relying on single score
**When to use:** When candidates are semantically similar but differ in key details
**Trade-offs:** More complex scoring but handles duplicate/similar papers better

**Example:**
```python
def weighted_rerank_score(signals: RerankSignals, task: str = "title_critical") -> float:
    """
    Compute overall relevance score from multi-signal evidence.
    
    Task modes adjust signal weights:
    - "title_critical": Heavily weight title match (for explicit mentions)
    - "author_critical": Prioritize author overlap (for citation contexts)
    - "semantic": Balance all signals (for vague queries)
    """
    
    if task == "title_critical":
        weights = {
            'title_similarity': 0.50,
            'author_overlap': 0.20,
            'method_consistency': 0.10,
            'finding_consistency': 0.10,
            'domain_alignment': 0.05,
            'venue_year_hints': 0.05,
        }
    elif task == "author_critical":
        weights = {
            'title_similarity': 0.25,
            'author_overlap': 0.40,
            'method_consistency': 0.10,
            'finding_consistency': 0.10,
            'domain_alignment': 0.10,
            'venue_year_hints': 0.05,
        }
    else:  # semantic
        weights = {
            'title_similarity': 0.20,
            'author_overlap': 0.15,
            'method_consistency': 0.20,
            'finding_consistency': 0.20,
            'domain_alignment': 0.15,
            'venue_year_hints': 0.10,
        }
    
    score = (
        signals.title_similarity * weights['title_similarity'] +
        signals.author_overlap * weights['author_overlap'] +
        signals.method_consistency * weights['method_consistency'] +
        signals.finding_consistency * weights['finding_consistency'] +
        signals.domain_alignment * weights['domain_alignment'] +
        signals.venue_year_hints * weights['venue_year_hints']
    )
    
    return score
```

**Why this matters for MRR@5:**
- Papers can be similar semantically but differ in authors/methods
- Handles duplicate title problem (56 duplicates in collection)
- Explicit signal tracking enables debugging and tuning

## Anti-Patterns to Avoid

### Anti-Pattern 1: Embedding Everything Without Structure

**What people do:** Concatenate all tweet text and all paper text, embed, compare
**Why it's wrong:** 
- Dilutes high-signal terms (title, authors) with noise
- No way to emphasize critical fields
- Embedding models have context limits - unstructured text wastes tokens

**Do this instead:** Use structured extraction → weighted input construction (Pattern 2)

### Anti-Pattern 2: Single-Stage Retrieval

**What people do:** Rely only on embedding similarity, no reranking
**Why it's wrong:**
- Embeddings capture general semantics but miss fine-grained signals (exact author match)
- Top-1 from embedding retrieval often wrong, but correct answer in top-20
- MRR@5 benefits hugely from reranking top-K

**Do this instead:** Always rerank top-K candidates with richer signals (Pattern 3)

### Anti-Pattern 3: Ignoring Extraction Failures

**What people do:** Silently fall back to raw text when extraction fails, don't log
**Why it's wrong:**
- Can't debug why extraction fails
- Don't know if problem is prompt, schema, or data quality
- Miss opportunities to improve extraction

**Do this instead:** Log extraction quality metrics, implement quality gates (Pattern 1)

### Anti-Pattern 4: Fixed Prompts Without Versioning

**What people do:** Hardcode prompts in code, iterate without tracking versions
**Why it's wrong:**
- Can't reproduce previous results
- Don't know which prompt changes helped vs hurt
- Difficult to do ablation studies

**Do this instead:** Version control prompt templates, config-driven prompt selection

```python
# Good: Versioned, tracked prompts
PROMPT_TEMPLATES = {
    "v1_baseline": "Extract evidence from tweet: {tweet}",
    "v2_structured": """Extract retrieval evidence from the social media post.
        Focus on: title mentions, author names, scientific terms, domain.
        Tweet: {tweet}""",
    "v3_examples": """Extract retrieval evidence from this scientific claim post.
        
        Example 1: [good extraction example]
        Example 2: [good extraction example]
        
        Now extract from:
        Tweet: {tweet}""",
}

class RetrievalConfig(BaseModel):
    extraction_prompt_version: str = "v3_examples"  # Track which version
```

### Anti-Pattern 5: Not Measuring Stage-Level Metrics

**What people do:** Only measure final MRR@5, don't track intermediate quality
**Why it's wrong:**
- Can't tell which stage is the bottleneck
- Optimize the wrong stage (e.g., improve reranking when retrieval recall is the issue)
- Waste time on changes that don't affect final metric

**Do this instead:** Track recall@K after retrieval, rerank quality, extraction confidence

```python
# Track per-stage metrics
metrics = {
    'extraction_confidence': 0.85,  # High confidence extraction
    'recall@50': 0.72,              # Correct paper in top-50 72% of time
    'recall@200': 0.89,             # Correct paper in top-200 89% of time
    'rerank_top1_confidence': 0.91, # Reranker confident about top-1
    'mrr@5': 0.67,                  # Final metric
}

# Identify bottleneck: low recall@50 → improve retrieval or query understanding
#                      high recall@200, low MRR@5 → improve reranking
```

## Integration Points & Data Flow

### Existing Integration: HuggingFace Dataset
**Current:** `load_dataset("sschellhammer/CT26_Task1_...")` for collection and splits
**Keep as-is:** Reliable, versioned, no need to change

### Existing Integration: Gemini API
**Current:** `genai.Client()` for embeddings and extraction
**Enhancements needed:**
- Add structured output validation for `RerankSignals`
- Implement retry logic for reranking calls (same as embedding batches)
- Monitor API costs per stage (extraction vs embedding vs reranking)

### New Integration: Reranker → Pipeline
**Addition:** `LLMReranker` class in `clef_retrieval/llm_reranker.py`

```python
# Pipeline integration point
def rank_from_query_embedding(
    tweet_text: str,
    query_embedding: np.ndarray,
    paper_embeddings: np.ndarray,
    metadata_rows: list[dict],
    top_k: int,
    reranker: LLMReranker | None = None,  # New parameter
) -> list[str]:
    """Enhanced ranking with optional LLM reranker."""
    
    # Stage 1: Dense retrieval (unchanged)
    pubkeys = [str(row.get("pubkey", "")) for row in metadata_rows]
    candidates = retrieve_top_pubkeys(
        query_embedding=query_embedding,
        paper_embeddings=paper_embeddings,
        pubkeys=pubkeys,
        k=top_k,
    )
    
    # Stage 2: Reranking (enhanced)
    if reranker is not None:
        # Use LLM reranker
        tweet_evidence = extract_tweet_evidence(tweet_text, service)  # Need service reference
        paper_evidences = [_metadata_to_evidence(metadata_rows, pk) for pk in candidates[:20]]
        reranked = reranker.rerank_candidates(tweet_evidence, paper_evidences)
        return ensure_top5([pk for pk, _ in reranked[:5]])
    else:
        # Fallback to lexical reranker (current behavior)
        metadata_by_pubkey = {str(row.get("pubkey", "")): row for row in metadata_rows}
        return stage2_rerank(
            tweet_text,
            candidates,
            rerank_fn=lambda text, cands: _lexical_rerank(text, cands, metadata_by_pubkey),
        )
```

## Scaling Considerations

| Query Volume | Architecture | Notes |
|--------------|--------------|-------|
| **Dev set (< 5K queries)** | Current: batch processing, cache predictions | Works well, ~5-10 min runtime |
| **Full train+dev (20K queries)** | Add: Parallel batch processing, chunk caching | Watch API rate limits |
| **Production (100K+ queries)** | Consider: Dedicated vector DB (Pinecone, Weaviate), separate reranking service | Out of scope for competition, but architecture supports it |

### Current Bottlenecks (Based on Code Analysis)

1. **Embedding API calls:** 10K papers × 1 embedding = ~10-15 min (one-time, cached)
2. **Query extraction:** N tweets × 1 LLM call = rate-limited by API (100 tweets/min typical)
3. **Reranking:** If doing LLM reranking, adds 1 LLM call per query batch

### Optimization Priorities

**Do now:**
- Cache predictions aggressively (already implemented ✓)
- Batch query extraction (already implemented ✓)
- Skip extraction if cached predictions exist (already implemented ✓)

**Do if reranker is slow:**
- Rerank only top-20 instead of top-200
- Batch multiple queries into single reranking prompt (if model supports)
- Use faster reranking model (Flash instead of Pro)

**Don't do yet:**
- Build custom vector index (NumPy cosine is fast enough for 10K papers)
- Optimize cold-start time (indexing is one-time cost)
- Distributed processing (dataset size doesn't justify)

## Recommended Project Structure

```
src/clef_retrieval/
├── __init__.py
├── config.py              # Configuration (keep as-is)
├── data.py                # Dataset loading (keep as-is)
├── schemas.py             # Pydantic schemas (ENHANCE: add confidence fields)
│
├── gemini_client.py       # ENHANCE: Improve extraction prompt
│   ├── extract_tweet_evidence()      # Add examples, field validation
│   └── build_embedding_input()       # Add weighting logic
│
├── paper_index.py         # ENHANCE: Richer paper representations
│   ├── build_paper_text()            # Optimize for retrieval matching
│   └── _build_metadata_row()         # Extract better keywords
│
├── retriever.py           # Keep as-is (works well)
│   ├── retrieve_top_pubkeys()
│   └── topk_indices()
│
├── llm_reranker.py        # NEW: LLM-based reranking
│   ├── LLMReranker
│   ├── rerank_candidates()
│   └── weighted_rerank_score()
│
├── reranker.py            # DEPRECATE or keep as fallback
│   └── _lexical_rerank()
│
└── pipeline.py            # ENHANCE: Integrate LLM reranker
    ├── build_query_embedding_text()  # Add quality gates
    ├── rank_from_query_embedding()   # Add reranker parameter
    └── predict_top5_with_embeddings()
```

### Structure Rationale

- **Separation of concerns:** Each module has clear responsibility
- **Backward compatibility:** New reranker doesn't break existing lexical fallback
- **Testability:** Each component can be tested in isolation
- **Caching boundaries:** Clear separation between offline (indexing) and online (prediction)

## Sources

- **Existing codebase analysis:** `/Users/mattias/.config/superpowers/worktrees/check-that-clef-2026/feat-clef-2026-retrieval-pipeline/`
- **Design specification:** `/Users/mattias/programmering/kth/check-that-clef-2026/docs/superpowers/specs/2026-03-30-clef-2026-source-retrieval-design.md`
- **Retrieval architecture patterns:** Two-stage retrieval + reranking is industry standard (ColBERT, DPR, SPLADE)
- **LLM reranking:** Established pattern in RAG systems (LlamaIndex, LangChain rerankers)
- **Multi-signal scoring:** Standard in search/recommender systems (learning-to-rank, ensemble methods)

**Confidence Assessment:**
- **Phase ordering:** HIGH - query understanding → reranking → paper enhancement is logical progression
- **Component boundaries:** HIGH - existing architecture is well-structured, enhancements fit cleanly
- **Expected impacts:** MEDIUM - MRR@5 estimates based on similar retrieval task patterns, actual results depend on data characteristics
- **Implementation complexity:** HIGH - code is clean, patterns are clear, Gemini API supports needed features

---
*Architecture research for: CLEF 2026 Task 1 Retrieval Quality Uplift*
*Researched: 2025-01-21*
*Based on: Existing codebase analysis + retrieval system design patterns*
