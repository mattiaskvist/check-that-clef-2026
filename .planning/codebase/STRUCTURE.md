# Codebase Structure

**Analysis Date:** 2025-03-31

## Directory Layout

```
check-that-clef-2026/
├── src/                            # Package source code
│   └── clef_retrieval/             # Main retrieval library
│       ├── __init__.py             # Package exports
│       ├── config.py               # Configuration schema
│       ├── schemas.py              # Data validation schemas
│       ├── data.py                 # Dataset loading utilities
│       ├── retriever.py            # Dense retrieval primitives
│       ├── evaluation.py           # Prediction validation
│       └── eda.py                  # Exploratory analysis utilities
├── tests/                          # Test suite
│       ├── test_config.py          # RetrievalConfig tests
│       ├── test_schemas.py         # TweetEvidence/PaperEvidence tests
│       ├── test_data_loading.py    # Dataset loading and normalization tests
│       ├── test_retriever.py       # Vector operation tests
│       └── test_evaluation.py      # Prediction validation tests
├── main.py                         # Entry point (stub)
├── scorer.py                       # MRR@5 evaluation script
├── pyproject.toml                  # Project metadata and dependencies
├── uv.lock                         # Dependency lock file
├── README.md                       # Quick start guide
└── docs/                           # Documentation
    └── superpowers/                # Planning documents
        ├── plans/                  # Implementation plans
        └── specs/                  # Specifications
```

## Directory Purposes

**`src/clef_retrieval/`:**
- Purpose: Core retrieval library for scientific paper evidence extraction and ranking
- Contains: Configuration, data schemas, data loading, vector operations, evaluation utilities
- Key files: `__init__.py` (exports), `config.py`, `schemas.py`, `data.py`, `retriever.py`

**`tests/`:**
- Purpose: Unit test suite for all library modules
- Contains: Test files mirroring src/ structure with test_ prefix
- Key files: Coverage includes config, schemas, data, retriever, evaluation (100 lines total)

**`docs/superpowers/`:**
- Purpose: GSD planning and specification documents
- Contains: Implementation plans and technical specifications
- Generated: Yes (created by GSD tooling)
- Committed: Yes

## Key File Locations

**Entry Points:**
- `main.py`: Primary application entry point (currently stub) - triggers pipeline orchestration
- `scorer.py`: MRR@5 evaluation script - computes ranking metric for predictions

**Configuration:**
- `pyproject.toml`: Project metadata, dependencies (datasets, pydantic, torch, transformers, google-genai, ollama)
- `src/clef_retrieval/config.py`: RetrievalConfig Pydantic model with embedding/reranking model names and search parameters

**Core Logic:**
- `src/clef_retrieval/schemas.py`: TweetEvidence (query metadata), PaperEvidence (document metadata)
- `src/clef_retrieval/data.py`: Dataset loading from HuggingFace Hub, author normalization
- `src/clef_retrieval/retriever.py`: L2 normalization, top-k selection using numpy
- `src/clef_retrieval/evaluation.py`: Prediction shape validation
- `src/clef_retrieval/eda.py`: Text length statistics for exploratory analysis

**Testing:**
- `tests/test_config.py`: RetrievalConfig default values validation
- `tests/test_schemas.py`: TweetEvidence and PaperEvidence instantiation and field validation
- `tests/test_data_loading.py`: Dataset loading with language/split parameters, author normalization edge cases
- `tests/test_retriever.py`: L2 normalization unit norm verification, top-k index ordering
- `tests/test_evaluation.py`: Prediction shape validation (must be exactly 5 per row)

## Naming Conventions

**Files:**
- `config.py`: Configuration schema files
- `schemas.py`: Pydantic model definitions
- `data.py`: Dataset loading and preprocessing utilities
- `retriever.py`: Core retrieval algorithm implementations
- `evaluation.py`: Metrics and validation logic
- `eda.py`: Exploratory data analysis utilities
- `test_*.py`: Test files with module-specific naming (test_config.py tests config.py)

**Directories:**
- `src/[package_name]/`: Library source code in snake_case
- `tests/`: Flat test directory with `test_*.py` files (pytest convention)
- `docs/superpowers/`: GSD-managed planning directory

**Functions:**
- Snake_case for all functions: `load_dataset()`, `normalize_authors()`, `topk_indices()`, `validate_prediction_shape()`
- Utility functions are lowercase with underscores

**Classes:**
- PascalCase for all classes: `RetrievalConfig`, `TweetEvidence`, `PaperEvidence`
- All are Pydantic BaseModel subclasses for data validation

**Variables:**
- snake_case for all variables and parameters: `top_k`, `embedding_model`, `pubkey`
- Field names in schemas match parameter names

## Where to Add New Code

**New Feature (retrieval enhancement):**
- Primary code: `src/clef_retrieval/retriever.py` - add new ranking/filtering functions
- Tests: `tests/test_retriever.py` - add unit tests for new retrieval logic
- Schemas: Update `src/clef_retrieval/schemas.py` if new evidence fields required

**New Module (e.g., re-ranker):**
- Implementation: Create `src/clef_retrieval/reranker.py` following module pattern
- Tests: Create `tests/test_reranker.py` with parallel test structure
- Config: Update `src/clef_retrieval/config.py` to add reranker configuration fields
- Init: Update `src/clef_retrieval/__init__.py` to export new public classes/functions

**Utilities (shared helpers):**
- Shared helpers: Add to appropriate existing module (`data.py`, `eda.py`, etc.)
- General utilities: Consider new `src/clef_retrieval/utils.py` only if crossing 3+ modules

**Dataset variants:**
- Data loading: Update `src/clef_retrieval/data.py` with new `load_*()` functions
- Dataset constant: Centralize DATASET_ID variable in data.py

## Special Directories

**`.planning/codebase/`:**
- Purpose: GSD codebase analysis documentation (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated: Yes (created by GSD mapper tool)
- Committed: Yes

**`.cache/clef_retrieval/`:**
- Purpose: Embedding cache directory specified in RetrievalConfig
- Generated: Yes (runtime creation by embedding models)
- Committed: No (in .gitignore)

**`.venv/`:**
- Purpose: Python virtual environment created by `uv sync`
- Generated: Yes (auto-created by uv package manager)
- Committed: No (in .gitignore)

**`docs/superpowers/`:**
- Purpose: GSD planning documents, implementation plans, and specifications
- Generated: Yes (created and maintained by GSD commands)
- Committed: Yes (version controlled)

## Module Dependency Map

```
main.py (entry point)
└── [No current imports - needs orchestration layer]

scorer.py (evaluation)
├── numpy (array operations)
├── datasets (HuggingFace loading)
└── [Not importing from src/clef_retrieval yet]

src/clef_retrieval/__init__.py
└── config.RetrievalConfig (exported public API)

src/clef_retrieval/config.py
└── pydantic.BaseModel, Field

src/clef_retrieval/schemas.py
└── pydantic.BaseModel, Field

src/clef_retrieval/data.py
├── datasets.load_dataset
├── collections.abc.Sequence
└── [Exports: normalize_authors, load_collection, load_language_split]

src/clef_retrieval/retriever.py
├── numpy (matrix operations, L2 norm)
└── [Exports: l2_normalize, topk_indices]

src/clef_retrieval/evaluation.py
├── collections.abc.Sequence
└── [Exports: validate_prediction_shape]

src/clef_retrieval/eda.py
├── statistics.mean, statistics.median
└── [Exports: summarize_lengths]
```

## Python Path Configuration

**pytest configuration** (`pyproject.toml`):
```
[tool.pytest.ini_options]
pythonpath = ["src"]
```

This allows tests to import directly from `clef_retrieval` namespace:
```python
from clef_retrieval.config import RetrievalConfig
from clef_retrieval.data import load_dataset
```

**Package location:** `src/clef_retrieval/` is installed as importable package via `uv sync`

---

*Structure analysis: 2025-03-31*
