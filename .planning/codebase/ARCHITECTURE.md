# Architecture

**Analysis Date:** 2025-03-31

## Pattern Overview

**Overall:** Modular utility library with layered separation of concerns

**Key Characteristics:**
- Functional programming approach with small, focused utility modules
- Pydantic-based schema validation for data structures
- Clear separation between data loading, configuration, retrieval, and evaluation
- No central application orchestration layer (currently stub-only in `main.py`)
- Test-driven module design with comprehensive test coverage

## Layers

**Configuration Layer:**
- Purpose: Define retrieval pipeline parameters using validated Pydantic models
- Location: `src/clef_retrieval/config.py`
- Contains: Pydantic BaseModel for RetrievalConfig with embedding/reranking models and parameters
- Depends on: pydantic
- Used by: Application orchestration (currently unused), evaluation scripts

**Data Layer:**
- Purpose: Handle dataset loading and normalization from HuggingFace Hub
- Location: `src/clef_retrieval/data.py`
- Contains: Functions for loading CLEF Task 1 dataset in different language splits, author normalization utilities
- Depends on: datasets library, collections.abc
- Used by: Retriever and evaluation pipelines

**Schema/Evidence Layer:**
- Purpose: Define structured data models for tweet and paper evidence with required fields
- Location: `src/clef_retrieval/schemas.py`
- Contains: TweetEvidence and PaperEvidence Pydantic models with extraction fields (keywords, methods, findings, domain terms)
- Depends on: pydantic
- Used by: Retrieval and ranking pipelines, test fixtures

**Retrieval Layer:**
- Purpose: Implement dense vector retrieval operations and ranking primitives
- Location: `src/clef_retrieval/retriever.py`
- Contains: L2 normalization for embedding vectors, efficient top-k selection using numpy
- Depends on: numpy
- Used by: Embedding-based retrieval pipelines

**Evaluation Layer:**
- Purpose: Validate prediction shapes and compute ranking metrics
- Location: `src/clef_retrieval/evaluation.py`
- Contains: Shape validation for top-5 predictions
- Depends on: collections.abc
- Used by: Scoring and evaluation scripts (standalone `scorer.py`)

**Exploratory Analysis Layer:**
- Purpose: Statistical utilities for quick data exploration
- Location: `src/clef_retrieval/eda.py`
- Contains: Text length statistics (mean, median) for dataset analysis
- Depends on: statistics library
- Used by: Notebooks and analysis scripts

**Scoring/Evaluation Script:**
- Purpose: Compute Mean Reciprocal Rank (MRR@5) metric from predictions
- Location: `scorer.py` (standalone script)
- Contains: MRR@5 calculation with dataset loading and label matching
- Depends on: numpy, datasets
- Used by: External evaluation pipelines

## Data Flow

**Dataset Collection & Retrieval Flow:**

1. Data is loaded from HuggingFace Hub via `load_dataset()` in `src/clef_retrieval/data.py`
2. Data is normalized (authors, fields) using utility functions
3. Paper evidence extracted and converted to embeddings using embedding_model from RetrievalConfig
4. Tweet evidence enriched with schema-based extraction (keywords, methods, findings) into TweetEvidence
5. Dense vectors L2-normalized via `l2_normalize()` in `src/clef_retrieval/retriever.py`
6. Top-K similar papers retrieved using `topk_indices()` in `src/clef_retrieval/retriever.py`
7. Top-N predictions ranked and formatted as 5-element lists
8. Predictions validated with `validate_prediction_shape()` in `src/clef_retrieval/evaluation.py`
9. MRR@5 score computed via `scorer()` in `scorer.py` by matching predictions against ground truth labels

**State Management:**

- Configuration state is immutable Pydantic model (`RetrievalConfig`) with validated defaults
- Dataset state is loaded once per split (train/dev) and cached by HuggingFace datasets library
- Embeddings are computed fresh per run (no caching in current implementation beyond `.cache/clef_retrieval`)
- Predictions are in-memory lists of lists (shape [num_queries, 5])

## Key Abstractions

**RetrievalConfig:**
- Purpose: Centralized configuration for embedding models, reranking models, and search parameters
- Examples: `src/clef_retrieval/config.py` (line 6-12)
- Pattern: Pydantic BaseModel with default values and field validation (ge, le)

**TweetEvidence & PaperEvidence:**
- Purpose: Type-safe structured containers for query and document metadata with optional extraction fields
- Examples: `src/clef_retrieval/schemas.py` (line 6-31)
- Pattern: Pydantic BaseModel with field defaults, required fields (pubkey, title, abstract), optional lists

**Dense Retrieval Primitives:**
- Purpose: Low-level vector operations for efficient similarity-based ranking
- Examples: `l2_normalize()` (line 8-15), `topk_indices()` (line 18-29) in `src/clef_retrieval/retriever.py`
- Pattern: Numpy-based functions with edge case handling (zero norm, empty arrays)

**Dataset Loaders:**
- Purpose: Abstraction over HuggingFace Hub dataset access with language/split parametrization
- Examples: `load_collection()`, `load_language_split()` in `src/clef_retrieval/data.py`
- Pattern: Functions encapsulating dataset IDs and subset names

## Entry Points

**Primary Entry Point (Stub):**
- Location: `main.py`
- Triggers: `uv run main.py`
- Responsibilities: Currently prints placeholder message; intended for orchestrating full pipeline

**Scoring Entry Point:**
- Location: `scorer.py`
- Triggers: Direct import or execution for evaluation
- Responsibilities: Loads predictions and dataset, computes MRR@5 metric against ground truth

**Test Entry Points:**
- Location: `tests/test_*.py` (5 test modules)
- Triggers: `pytest` command via uv
- Responsibilities: Unit testing individual modules in isolation

## Error Handling

**Strategy:** Assertions and exception propagation

**Patterns:**
- `scorer.py` uses assertions for language/split validation and prediction shape validation
- `l2_normalize()` handles zero-norm vectors by replacing with 1.0 to avoid division errors
- `topk_indices()` returns empty list for edge cases (empty similarities or k <= 0)
- `normalize_authors()` safely handles None, strings, and sequences with fallback to empty string
- Pydantic models provide validation errors for type mismatches on config instantiation

## Cross-Cutting Concerns

**Logging:** None detected - use print statements only

**Validation:** Pydantic BaseModel validation on config/schemas; manual assertions in scorer.py

**Authentication:** None required - HuggingFace Hub dataset is public, accessed via datasets library

**Caching:** HuggingFace datasets library handles automatic caching; embedding cache configured in `RetrievalConfig.cache_dir`

---

*Architecture analysis: 2025-03-31*
