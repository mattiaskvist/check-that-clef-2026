# Coding Conventions

**Analysis Date:** 2025-01-14

## Naming Patterns

**Files:**
- Lowercase with underscores: `config.py`, `data.py`, `retriever.py`, `evaluation.py`, `schemas.py`
- Module package: `clef_retrieval/` (lowercase with underscores)
- Test files: `test_<module>.py` (e.g., `test_config.py`, `test_data_loading.py`)

**Functions:**
- Snake_case: `normalize_authors()`, `load_collection()`, `l2_normalize()`, `topk_indices()`, `validate_prediction_shape()`, `summarize_lengths()`
- Descriptive verb-based names that indicate action: `load_*`, `normalize_*`, `validate_*`, `summarize_*`
- Convention: utility/helper functions named as actions

**Variables:**
- Snake_case throughout: `embedding_model`, `rerank_model`, `top_k`, `top_n`, `cache_dir`
- Descriptive names indicating content: `authors`, `labels`, `mrr_scores`, `norms`, `similarities`
- Type hints preferred on function parameters

**Types/Classes:**
- PascalCase: `RetrievalConfig`, `TweetEvidence`, `PaperEvidence`
- Pydantic BaseModel pattern used for data validation classes
- Field descriptors with defaults: `Field(default="value")`, `Field(default_factory=list)`

**Constants:**
- UPPERCASE: `DATASET_ID = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"`

## Code Style

**Formatting:**
- Ruff is installed as linter (`ruff>=0.15.7` in `pyproject.toml`)
- No explicit formatter configured (ruff provides linting only)
- Python 3.14+ required (`requires-python = ">=3.14"`)

**Linting:**
- Ruff linter configured via `pyproject.toml`
- Cache directory: `.ruff_cache/`
- No explicit configuration block in `pyproject.toml` yet

**Type Hints:**
- Function parameters typed: `def normalize_authors(authors: object) -> str:`
- Return types specified: `-> str`, `-> list[int]`, `-> dict[str, float]`, `-> bool`
- Modern union syntax with `|` not used; using older `object` or specific types
- Collection types use built-in generics: `list[str]`, `dict[str, float]`, `list[int]`

## Import Organization

**Order:**
1. `from __future__ import annotations` (when needed for forward references)
2. Standard library imports: `import numpy as np`, `from collections.abc import Sequence`, `from statistics import mean, median`
3. Third-party imports: `from datasets import load_dataset`, `from pydantic import BaseModel, Field`
4. Relative imports: `from .config import RetrievalConfig`

**Path Aliases:**
- No explicit path aliases configured
- Relative imports used within package: `from .config import`
- Tests configured with `pythonpath = ["src"]` in `pyproject.toml` to allow `from clef_retrieval import` statements

**Example from `src/clef_retrieval/__init__.py`:**
```python
from .config import RetrievalConfig

__all__ = ["RetrievalConfig"]
```

## Error Handling

**Patterns:**
- No try-except blocks in core modules (verified in `src/clef_retrieval/`)
- Validation via Pydantic models: `RetrievalConfig`, `TweetEvidence`, `PaperEvidence` with `Field()` constraints
- Explicit assertions in `scorer.py`:
  ```python
  assert lang in ["de", "en", "fr"], "You need to provide a correct language parameter..."
  assert split in ["train", "dev"], "You need to provide a correct split parameter..."
  assert len(preds) == 5, "exactly 5 predictions per query should be provided"
  ```
- Functional validation pattern: `validate_prediction_shape()` returns boolean instead of raising

**Guidelines for new code:**
- Use Pydantic models for structured data validation over manual checks
- Return boolean validation results for non-critical checks
- Use assertions for parameter validation when expectations are strict
- Avoid silent failures; prefer explicit assertions over conditional returns

## Logging

**Framework:** Not detected. Uses `print()` for user-facing output (see `main.py`).

**Patterns:**
- Minimal logging in production code
- Docstrings provide context instead of inline logging
- No logging imports found in `src/clef_retrieval/`

## Comments

**When to Comment:**
- Docstrings on modules (present in all files): `"""Configuration for the retrieval pipeline."""`
- Docstrings on functions where complexity warrants it: See `scorer()` function with full Args/Returns documentation
- No inline comments found; code is self-documenting

**JSDoc/TSDoc:**
- Not applicable (Python project)
- Docstring convention observed: Module-level, function-level docstrings
- Function docstring example from `scorer.py`:
  ```python
  """
  Compute MRR@5 given top 5 predictions, language and split information.

  Args:
      top5_preds: list of lists (shape [num_queries, 5])
      lang: str to indicate the language (de, en, fr)
      split: str to indicate the data split (train, dev)

  Returns:
      float: MRR@5 score
  """
  ```

## Function Design

**Size:** Minimal, focused functions. Range: 3-15 lines typical.
- `l2_normalize()`: 8 lines - single responsibility (matrix normalization)
- `topk_indices()`: 9 lines - single responsibility (retrieve top-k indices)
- `normalize_authors()`: 7 lines - handles author normalization with type dispatch

**Parameters:**
- Few parameters preferred; typically 1-3 parameters
- Type hints required
- Sequence[T] used for flexible input types: `Sequence[object]`, `Sequence[Sequence[object]]`

**Return Values:**
- Explicit types specified
- Return `[]` for empty lists: `return []`
- Return `""` for empty strings: `return ""`
- Return `{}` for empty dicts: typical Pydantic Field default pattern

**Example from `src/clef_retrieval/data.py`:**
```python
def normalize_authors(authors: object) -> str:
    if isinstance(authors, str):
        return authors.strip()
    if isinstance(authors, Sequence):
        values = [str(author).strip() for author in authors]
        return "; ".join(value for value in values if value)
    return ""
```

## Module Design

**Exports:**
- Barrel file pattern: `src/clef_retrieval/__init__.py` defines `__all__`:
  ```python
  __all__ = ["RetrievalConfig"]
  ```
- Only essential exports exposed; internal utilities remain in submodules

**Barrel Files:**
- Used in `src/clef_retrieval/__init__.py` to expose `RetrievalConfig`
- Pattern: Public API in `__init__.py`, implementation in submodules

**Module Purposes:**
- `config.py`: Configuration models and defaults
- `schemas.py`: Data validation schemas (Pydantic models)
- `data.py`: Dataset loading and data preparation functions
- `retriever.py`: Dense retrieval operations (embedding normalization, top-k selection)
- `evaluation.py`: Evaluation and validation helpers
- `eda.py`: Exploratory data analysis utilities

---

*Convention analysis: 2025-01-14*
