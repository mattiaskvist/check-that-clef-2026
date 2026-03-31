# CLEF 2026 CheckThat Task 1 Design: Source Retrieval for Scientific Web Claims

## Problem
Given a social media post (English, German, or French) that references a scientific paper without a URL, retrieve the correct paper from a fixed candidate pool (10,000 papers), optimized for **MRR@5**.

## Agreed approach
- Retrieval architecture: **two-stage** system
- Embedding model: **Gemini Embeddings 2**
- LLM for query/paper structuring and reranking: **Gemini 3.1 Flash-Lite Preview**
- Paper embedding content: **title + abstract + authors + highlights/keywords**
- Priority: **maximize leaderboard performance**

## EDA summary (ground truth context)
- Collection size: 10,000 papers
- Collection fields: `pubkey`, `title`, `abstract`, `venue`, `authors`
- Completeness: title/abstract/pubkey are fully populated
- Paper text length:
  - title mean ~13 words
  - abstract mean ~232 words (long tail to ~452 words at p95)
- Query datasets:
  - English train/dev: 14977 / 3905
  - German train/dev: 1460 / 386
  - French train/dev: 2807 / 702
- Tweet lengths are short (~29–34 words mean), so extraction/enrichment is important.
- 56 duplicate normalized titles exist in collection, requiring disambiguation beyond title string.

## Architecture

### Stage 0: Offline paper preparation and indexing
1. Load collection.
2. For each paper, build a structured representation:
   - Raw fields: title, authors, abstract, venue
   - Extracted fields (LLM-generated): keywords, method_terms, finding_terms, domain_terms, short highlights
3. Build a dense embedding input (`paper_text_for_embedding`) that combines:
   - Title
   - Authors
   - Abstract
   - LLM highlights/keywords
4. Embed all papers with Gemini Embeddings 2.
5. Build ANN index over paper embeddings.
6. Persist:
   - embeddings
   - index artifacts
   - mapping from index IDs to `pubkey` and metadata

### Stage 1: Online query enrichment + dense retrieval
1. Input: tweet text.
2. Run Gemini 3.1 Flash-Lite with structured output schema to extract retrieval evidence.
3. Build `query_text_for_embedding` from original tweet + extracted fields.
4. Embed query with Gemini Embeddings 2.
5. ANN search to retrieve top-K candidates (target K in tuning grid, e.g. 100/200/400).

### Stage 2: LLM reranking
1. Feed query evidence and top-K candidate metadata to Gemini 3.1 Flash-Lite.
2. Score candidates with emphasis on:
   - title proximity (exact/fuzzy)
   - author overlap/hints
   - method/finding/domain consistency
   - venue/year/context clues if present
3. Return ranked top-5 `pubkey` predictions for scoring.

## Structured schemas

### Tweet enrichment schema
- `claim_summary`: concise restatement of the scientific claim
- `language`: detected language (`en`, `de`, `fr`, or unknown)
- `candidate_title_mentions[]`: possible paper title fragments
- `candidate_authors[]`: author names/handles/aliases
- `keywords[]`: high-signal topical terms
- `method_terms[]`: methods, datasets, trial types, analysis terms
- `finding_terms[]`: result-oriented terms/effects
- `domain_terms[]`: field/application anchors
- `time_or_venue_hints[]`: years, journals, conference hints
- `negative_constraints[]`: explicit mismatch constraints
- `query_text_for_embedding`: retrieval-ready synthesized query text

### Paper enrichment schema
- Raw: `title`, `authors`, `abstract`, `venue`
- Extracted: `keywords[]`, `method_terms[]`, `finding_terms[]`, `domain_terms[]`, `highlights[]`
- `paper_text_for_embedding`: compact information-dense summary string for embedding

### Reranker output schema
- `ranked_pubkeys[]` (top 5)
- `scores[]` (normalized relevance scores)
- `rationale_short[]` (brief evidence snippets per selected candidate)

## Multilingual strategy
- Keep tweet language as-is for first-pass retrieval.
- Avoid mandatory translation step initially; rely on multilingual/cross-lingual semantic embedding behavior.
- If multilingual mismatch appears during tuning, optionally add translation-augmented query text as a secondary variant.

## Duplicate-title disambiguation strategy
For candidates with same/similar normalized titles, prioritize:
1. author overlap
2. method/finding consistency
3. domain-term agreement
4. venue and contextual hints

This addresses known title collisions in the collection.

## Error handling and robustness
- If structured extraction fails or is partial:
  - fallback to minimally processed original tweet text for embedding
  - still run retrieval/reranking pipeline
- Never silently drop samples; log failures with sample identifiers.
- Cache all expensive outputs:
  - paper enrichments
  - paper embeddings
  - tweet enrichments
  - query embeddings

## Evaluation plan
- Primary metric: **MRR@5** using provided scorer.
- Evaluate per language and aggregate.
- Ablation ladder:
  1. Dense-only baseline (tweet raw text vs paper raw text)
  2. + tweet enrichment
  3. + paper enrichment
  4. + LLM reranker
  5. K and prompt tuning
- Report:
  - MRR@5 (overall + en/de/fr)
  - cost and runtime per 1k tweets
  - error buckets (wrong domain, title-confusion, author mismatch, language drift)

## Implementation boundaries (in scope / out of scope)
### In scope
- End-to-end reproducible CLI pipeline:
  - `build-index`
  - `predict`
  - `evaluate`
- Configurable model names, K values, and caching paths.
- Deterministic structured outputs where possible.

### Out of scope (for this cycle)
- Training custom embedding models
- External web retrieval
- Complex late-interaction rerankers requiring heavy model infrastructure

## Risks and mitigations
- **LLM extraction hallucination**: constrain with strict schema + low temperature + reranker grounding against candidates.
- **Cost blow-up**: cache aggressively and rerank only top-K.
- **Cross-lingual misses**: monitor per-language MRR and add translation-augmented query variant if needed.
- **Duplicate title confusion**: use author/method/finding features in reranker.

## Deliverables
1. EDA script + dataset profile output
2. Offline index builder for enriched paper embeddings
3. Tweet enrichment and retrieval pipeline
4. Top-K reranker producing top-5 pubkeys
5. Evaluation CLI with scorer integration and ablation outputs
