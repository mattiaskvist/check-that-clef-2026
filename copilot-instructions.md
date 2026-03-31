<!-- GSD:project-start source:PROJECT.md -->
## Project

**CLEF 2026 CheckThat Task 1 Retrieval Improvement**

This project focuses on improving a multilingual source-retrieval pipeline for CLEF 2026 CheckThat Task 1. Given a social media post with an implicit scientific paper reference, the system should retrieve the correct paper from a candidate pool. The immediate goal is to improve quality over the current in-repo baseline while keeping iteration speed practical.

**Core Value:** Improve MRR@5 on the dev split over the repository's current baseline for Task 1 source retrieval.

### Constraints

- **Task Scope**: CLEF 2026 CheckThat Task 1 only — project focus is constrained to source retrieval for scientific web claims
- **Model Direction**: Use Gemini Embeddings 2 plus LLM-assisted query understanding — chosen approach should stay consistent
- **Evaluation Target**: Dev MRR@5 vs in-repo baseline — progress must be measured against this metric
- **Data Boundary**: Existing CLEF dataset splits only — no new external dataset ingestion for this initiative
- **Cost Awareness**: Initial spend cap around 200 SEK — improvements should be mindful of API usage and query-time overhead
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.14+ - All application code, ML models, and data processing
## Runtime
- Python 3.14 or higher (as specified in `pyproject.toml`)
- UV (modern Python package manager)
- Lockfile: `uv.lock` present and maintained
## Frameworks
- Pydantic 2.12.5+ - Data validation and schema definition (`src/clef_retrieval/schemas.py`, `src/clef_retrieval/config.py`)
- Datasets 4.8.4+ - HuggingFace dataset loading and management (`src/clef_retrieval/data.py`)
- PyTorch 2.11.0+ - Deep learning framework for model inference
- Transformers 5.3.0+ - HuggingFace transformers library for NLP models
- Ollama 0.6.1+ - Local LLM inference engine (`OsvaldsTry.py` uses `ollama.pull()` and embedding generation)
- Google GenAI 1.68.0+ - Google Gemini API integration (`geminitry.py` uses `google.genai`)
- Pytest 8.4.2+ - Test framework and runner
- Ruff 0.15.7+ - Fast Python linter and formatter
- Python-dotenv 1.2.2 - Environment variable loading (`geminitry.py` uses `load_dotenv()`)
- NumPy 2.4.3+ - Numerical computing (dense retrieval operations in `src/clef_retrieval/retriever.py`)
- IPykernel 7.2.0+ - Jupyter notebook kernel support (for `CT26_Task1_baseline.ipynb`)
## Key Dependencies
- `datasets>=4.8.4` - Core dependency for loading CT26 dataset from HuggingFace Hub
- `google-genai>=1.68.0` - Google Gemini AI API client
- `ollama>=0.6.1` - Local LLM inference (Ollama client)
- `pydantic>=2.12.5` - Data validation for configuration and schemas
- `huggingface-hub>=1.7.2` - HuggingFace Hub API (dependency of `datasets`)
- `torch>=2.11.0` - PyTorch for model inference
- `transformers>=5.3.0` - HuggingFace model library for NLP tasks
## Configuration
- Configuration via `src/clef_retrieval/config.py` (`RetrievalConfig` Pydantic model)
- Environment variables (via `python-dotenv`):
- `pyproject.toml` - Python project configuration (PEP 518/PEP 621 standard)
## Platform Requirements
- macOS/Linux/Windows with Python 3.14+
- `brew install uv` (for macOS) or equivalent for other platforms
- HuggingFace authentication: `uv run hf auth login`
- Google Gemini API key (stored in `.env` via `python-dotenv`)
- Optional: Ollama installed locally for local LLM inference
- Python 3.14+ runtime
- Network access to HuggingFace Hub API
- Network access to Google Gemini API (for `geminitry.py` usage)
- Optional: Local Ollama installation (for `OsvaldsTry.py` usage)
- Cache directory writeable (default: `.cache/clef_retrieval/`)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Lowercase with underscores: `config.py`, `data.py`, `retriever.py`, `evaluation.py`, `schemas.py`
- Module package: `clef_retrieval/` (lowercase with underscores)
- Test files: `test_<module>.py` (e.g., `test_config.py`, `test_data_loading.py`)
- Snake_case: `normalize_authors()`, `load_collection()`, `l2_normalize()`, `topk_indices()`, `validate_prediction_shape()`, `summarize_lengths()`
- Descriptive verb-based names that indicate action: `load_*`, `normalize_*`, `validate_*`, `summarize_*`
- Convention: utility/helper functions named as actions
- Snake_case throughout: `embedding_model`, `rerank_model`, `top_k`, `top_n`, `cache_dir`
- Descriptive names indicating content: `authors`, `labels`, `mrr_scores`, `norms`, `similarities`
- Type hints preferred on function parameters
- PascalCase: `RetrievalConfig`, `TweetEvidence`, `PaperEvidence`
- Pydantic BaseModel pattern used for data validation classes
- Field descriptors with defaults: `Field(default="value")`, `Field(default_factory=list)`
- UPPERCASE: `DATASET_ID = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"`
## Code Style
- Ruff is installed as linter (`ruff>=0.15.7` in `pyproject.toml`)
- No explicit formatter configured (ruff provides linting only)
- Python 3.14+ required (`requires-python = ">=3.14"`)
- Ruff linter configured via `pyproject.toml`
- Cache directory: `.ruff_cache/`
- No explicit configuration block in `pyproject.toml` yet
- Function parameters typed: `def normalize_authors(authors: object) -> str:`
- Return types specified: `-> str`, `-> list[int]`, `-> dict[str, float]`, `-> bool`
- Modern union syntax with `|` not used; using older `object` or specific types
- Collection types use built-in generics: `list[str]`, `dict[str, float]`, `list[int]`
## Import Organization
- No explicit path aliases configured
- Relative imports used within package: `from .config import`
- Tests configured with `pythonpath = ["src"]` in `pyproject.toml` to allow `from clef_retrieval import` statements
## Error Handling
- No try-except blocks in core modules (verified in `src/clef_retrieval/`)
- Validation via Pydantic models: `RetrievalConfig`, `TweetEvidence`, `PaperEvidence` with `Field()` constraints
- Explicit assertions in `scorer.py`:
- Functional validation pattern: `validate_prediction_shape()` returns boolean instead of raising
- Use Pydantic models for structured data validation over manual checks
- Return boolean validation results for non-critical checks
- Use assertions for parameter validation when expectations are strict
- Avoid silent failures; prefer explicit assertions over conditional returns
## Logging
- Minimal logging in production code
- Docstrings provide context instead of inline logging
- No logging imports found in `src/clef_retrieval/`
## Comments
- Docstrings on modules (present in all files): `"""Configuration for the retrieval pipeline."""`
- Docstrings on functions where complexity warrants it: See `scorer()` function with full Args/Returns documentation
- No inline comments found; code is self-documenting
- Not applicable (Python project)
- Docstring convention observed: Module-level, function-level docstrings
- Function docstring example from `scorer.py`:
## Function Design
- `l2_normalize()`: 8 lines - single responsibility (matrix normalization)
- `topk_indices()`: 9 lines - single responsibility (retrieve top-k indices)
- `normalize_authors()`: 7 lines - handles author normalization with type dispatch
- Few parameters preferred; typically 1-3 parameters
- Type hints required
- Sequence[T] used for flexible input types: `Sequence[object]`, `Sequence[Sequence[object]]`
- Explicit types specified
- Return `[]` for empty lists: `return []`
- Return `""` for empty strings: `return ""`
- Return `{}` for empty dicts: typical Pydantic Field default pattern
## Module Design
- Barrel file pattern: `src/clef_retrieval/__init__.py` defines `__all__`:
- Only essential exports exposed; internal utilities remain in submodules
- Used in `src/clef_retrieval/__init__.py` to expose `RetrievalConfig`
- Pattern: Public API in `__init__.py`, implementation in submodules
- `config.py`: Configuration models and defaults
- `schemas.py`: Data validation schemas (Pydantic models)
- `data.py`: Dataset loading and data preparation functions
- `retriever.py`: Dense retrieval operations (embedding normalization, top-k selection)
- `evaluation.py`: Evaluation and validation helpers
- `eda.py`: Exploratory data analysis utilities
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Functional programming approach with small, focused utility modules
- Pydantic-based schema validation for data structures
- Clear separation between data loading, configuration, retrieval, and evaluation
- No central application orchestration layer (currently stub-only in `main.py`)
- Test-driven module design with comprehensive test coverage
## Layers
- Purpose: Define retrieval pipeline parameters using validated Pydantic models
- Location: `src/clef_retrieval/config.py`
- Contains: Pydantic BaseModel for RetrievalConfig with embedding/reranking models and parameters
- Depends on: pydantic
- Used by: Application orchestration (currently unused), evaluation scripts
- Purpose: Handle dataset loading and normalization from HuggingFace Hub
- Location: `src/clef_retrieval/data.py`
- Contains: Functions for loading CLEF Task 1 dataset in different language splits, author normalization utilities
- Depends on: datasets library, collections.abc
- Used by: Retriever and evaluation pipelines
- Purpose: Define structured data models for tweet and paper evidence with required fields
- Location: `src/clef_retrieval/schemas.py`
- Contains: TweetEvidence and PaperEvidence Pydantic models with extraction fields (keywords, methods, findings, domain terms)
- Depends on: pydantic
- Used by: Retrieval and ranking pipelines, test fixtures
- Purpose: Implement dense vector retrieval operations and ranking primitives
- Location: `src/clef_retrieval/retriever.py`
- Contains: L2 normalization for embedding vectors, efficient top-k selection using numpy
- Depends on: numpy
- Used by: Embedding-based retrieval pipelines
- Purpose: Validate prediction shapes and compute ranking metrics
- Location: `src/clef_retrieval/evaluation.py`
- Contains: Shape validation for top-5 predictions
- Depends on: collections.abc
- Used by: Scoring and evaluation scripts (standalone `scorer.py`)
- Purpose: Statistical utilities for quick data exploration
- Location: `src/clef_retrieval/eda.py`
- Contains: Text length statistics (mean, median) for dataset analysis
- Depends on: statistics library
- Used by: Notebooks and analysis scripts
- Purpose: Compute Mean Reciprocal Rank (MRR@5) metric from predictions
- Location: `scorer.py` (standalone script)
- Contains: MRR@5 calculation with dataset loading and label matching
- Depends on: numpy, datasets
- Used by: External evaluation pipelines
## Data Flow
- Configuration state is immutable Pydantic model (`RetrievalConfig`) with validated defaults
- Dataset state is loaded once per split (train/dev) and cached by HuggingFace datasets library
- Embeddings are computed fresh per run (no caching in current implementation beyond `.cache/clef_retrieval`)
- Predictions are in-memory lists of lists (shape [num_queries, 5])
## Key Abstractions
- Purpose: Centralized configuration for embedding models, reranking models, and search parameters
- Examples: `src/clef_retrieval/config.py` (line 6-12)
- Pattern: Pydantic BaseModel with default values and field validation (ge, le)
- Purpose: Type-safe structured containers for query and document metadata with optional extraction fields
- Examples: `src/clef_retrieval/schemas.py` (line 6-31)
- Pattern: Pydantic BaseModel with field defaults, required fields (pubkey, title, abstract), optional lists
- Purpose: Low-level vector operations for efficient similarity-based ranking
- Examples: `l2_normalize()` (line 8-15), `topk_indices()` (line 18-29) in `src/clef_retrieval/retriever.py`
- Pattern: Numpy-based functions with edge case handling (zero norm, empty arrays)
- Purpose: Abstraction over HuggingFace Hub dataset access with language/split parametrization
- Examples: `load_collection()`, `load_language_split()` in `src/clef_retrieval/data.py`
- Pattern: Functions encapsulating dataset IDs and subset names
## Entry Points
- Location: `main.py`
- Triggers: `uv run main.py`
- Responsibilities: Currently prints placeholder message; intended for orchestrating full pipeline
- Location: `scorer.py`
- Triggers: Direct import or execution for evaluation
- Responsibilities: Loads predictions and dataset, computes MRR@5 metric against ground truth
- Location: `tests/test_*.py` (5 test modules)
- Triggers: `pytest` command via uv
- Responsibilities: Unit testing individual modules in isolation
## Error Handling
- `scorer.py` uses assertions for language/split validation and prediction shape validation
- `l2_normalize()` handles zero-norm vectors by replacing with 1.0 to avoid division errors
- `topk_indices()` returns empty list for edge cases (empty similarities or k <= 0)
- `normalize_authors()` safely handles None, strings, and sequences with fallback to empty string
- Pydantic models provide validation errors for type mismatches on config instantiation
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
