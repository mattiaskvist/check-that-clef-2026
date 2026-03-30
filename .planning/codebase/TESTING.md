# Testing Patterns

**Analysis Date:** 2025-01-14

## Test Framework

**Runner:**
- pytest 9.0.2+
- Config file: `pyproject.toml`
- Configuration in `pyproject.toml`:
  ```ini
  [tool.pytest.ini_options]
  pythonpath = ["src"]
  ```

**Assertion Library:**
- pytest's built-in assertions: `assert`, `assert condition`
- NumPy assertions for numerical testing: `np.isclose()`, `np.array_equal()`

**Run Commands:**
```bash
python -m pytest tests/                    # Run all tests
python -m pytest tests/ -v                 # Verbose output with test names
python -m pytest tests/test_config.py      # Run specific test file
```

**Current Status:** All 10 tests passing as of last run

## Test File Organization

**Location:**
- Separate directory: `tests/` at repository root
- Co-located with `src/`, not alongside source files

**Naming:**
- Pattern: `test_<module_name>.py`
- Examples: `test_config.py`, `test_data_loading.py`, `test_evaluation.py`, `test_retriever.py`, `test_schemas.py`
- Test functions prefixed with `test_`: `test_default_config_values()`, `test_normalize_authors_handles_lists_and_strings()`

**Directory Structure:**
```
tests/
├── test_config.py              # 11 lines
├── test_data_loading.py        # 39 lines (largest)
├── test_evaluation.py          # 9 lines
├── test_retriever.py           # 18 lines
└── test_schemas.py             # 23 lines

Total: 5 test modules, 100 lines, 10 test functions
```

## Test Structure

**Test Function Pattern:**

Simple assertion-based tests dominate the codebase:

```python
def test_default_config_values():
    cfg = RetrievalConfig()
    assert cfg.embedding_model == "models/gemini-embedding-2-preview"
    assert cfg.rerank_model == "gemini-3.1-flash-lite-preview"
    assert cfg.top_k == 200
    assert cfg.cache_dir == ".cache/clef_retrieval"
```

**Test Organization Pattern from `test_data_loading.py`:**
```python
from clef_retrieval.data import load_collection, load_language_split, normalize_authors


def test_normalize_authors_handles_lists_and_strings():
    # Test data setup inline
    assert normalize_authors(["A. One", "B. Two"]) == "A. One; B. Two"
    assert normalize_authors(" C. Three ") == "C. Three"
    assert normalize_authors(None) == ""
```

**Patterns:**
- No explicit setup/teardown methods
- Test data created inline (no fixtures)
- Single assertion per test concept (though multiple assertions within one test allowed)
- Mocking handled via pytest's `monkeypatch` fixture

## Mocking

**Framework:** pytest built-in `monkeypatch` fixture

**Pattern from `test_data_loading.py`:**
```python
def test_load_collection_uses_collection_subset(monkeypatch):
    calls = {}

    def fake_load_dataset(dataset_id, subset):
        calls["args"] = (dataset_id, subset)
        return {"collection": ["paper-1", "paper-2"]}

    monkeypatch.setattr("clef_retrieval.data.load_dataset", fake_load_dataset)

    assert load_collection() == ["paper-1", "paper-2"]
    assert calls["args"] == (
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        "collection",
    )
```

**Approach:**
- Function-level mocking using `monkeypatch.setattr()`
- Mock function captures call arguments in a dictionary
- Return value constructed as test data
- No external libraries (unittest.mock, pytest-mock) used

**What to Mock:**
- External service calls: `load_dataset()` calls mocked in data loading tests
- API calls that require network access
- Heavy dependencies that slow down tests

**What NOT to Mock:**
- Pydantic models used for validation (test their actual behavior)
- Pure utility functions with no side effects
- NumPy operations (test with real arrays)

## Fixtures and Factories

**Test Data:**
- No fixture files or conftest.py found
- Test data created inline within test functions
- Examples from `test_retriever.py`:
  ```python
  def test_l2_normalize_scales_rows_to_unit_norm():
      matrix = np.array([[3.0, 4.0], [0.0, 0.0]])
      normalized = l2_normalize(matrix)
      assert np.isclose(np.linalg.norm(normalized[0]), 1.0)
      assert np.array_equal(normalized[1], np.array([0.0, 0.0]))
  ```

- Pydantic model instantiation as test data from `test_schemas.py`:
  ```python
  def test_paper_evidence_has_required_fields():
      paper = PaperEvidence(
          pubkey="1",
          title="Title",
          authors="A",
          abstract="B",
          paper_text_for_embedding="T",
      )
      assert paper.pubkey == "1"
  ```

**Location:**
- No separate fixtures directory
- Test data is local to each test function
- Mocked functions defined inline in test functions

## Coverage

**Requirements:** Not enforced (no pytest-cov plugin installed)

**Current Coverage:** Not measured

**View Coverage:** Not available without additional tool installation

**Recommendation for measuring coverage:**
- Add `pytest-cov` to dev dependencies
- Run: `python -m pytest tests/ --cov=src/clef_retrieval --cov-report=html`
- Currently untested areas: `src/clef_retrieval/eda.py` (no test file for eda module)

## Test Types

**Unit Tests:**
- Scope: Individual functions in isolation
- Approach: Direct function calls with simple inputs, assertion on outputs
- Examples:
  - `test_default_config_values()`: Tests config model defaults
  - `test_l2_normalize_scales_rows_to_unit_norm()`: Tests normalization behavior
  - `test_topk_indices_returns_descending_similarity()`: Tests retrieval ranking

**Integration Tests:**
- Scope: Not present in current test suite
- Could test: Data loading with mocked datasets, config + retrieval pipeline

**E2E Tests:**
- Framework: Not used
- Notebooks exist (`CT26_Task1_baseline.ipynb`) but not automated E2E tests

## Common Patterns

**Async Testing:**
- Not applicable (synchronous functions only)

**Error Testing:**
- Assertions used instead of exception testing
- Example of validation testing from `test_evaluation.py`:
  ```python
  def test_validate_prediction_shape_rejects_rows_with_wrong_length():
      assert not validate_prediction_shape([["1", "2", "3", "4"], ["a", "b", "c", "d", "e"]])
  ```
- No explicit exception assertions; validation returns boolean

**Numerical Testing:**
- NumPy assertions for floating-point comparisons:
  ```python
  assert np.isclose(np.linalg.norm(normalized[0]), 1.0)  # Tolerates floating-point error
  assert np.array_equal(normalized[1], np.array([0.0, 0.0]))
  ```

**Argument Capture Pattern (for mocking):**
```python
calls = {}
def fake_function(arg1, arg2):
    calls["args"] = (arg1, arg2)
    return expected_result
monkeypatch.setattr("module.function", fake_function)
# Later: assert calls["args"] == (expected_arg1, expected_arg2)
```

## Test Modules Reference

**`tests/test_config.py` (11 lines):**
- Tests: `RetrievalConfig` model initialization and defaults
- Coverage: 1 test function

**`tests/test_schemas.py` (23 lines):**
- Tests: `TweetEvidence` and `PaperEvidence` Pydantic models
- Coverage: 2 test functions, default field values, required fields

**`tests/test_evaluation.py` (9 lines):**
- Tests: `validate_prediction_shape()` validation function
- Coverage: 2 test functions (accept valid, reject invalid shapes)

**`tests/test_retriever.py` (18 lines):**
- Tests: `l2_normalize()` and `topk_indices()` retrieval functions
- Coverage: 2 test functions, handles edge cases (zero norms, empty arrays)
- Uses NumPy assertions

**`tests/test_data_loading.py` (39 lines - largest):**
- Tests: `normalize_authors()`, `load_collection()`, `load_language_split()`
- Coverage: 3 test functions
- Uses `monkeypatch` to mock `load_dataset()` from Hugging Face Datasets library
- Verifies function arguments and return values

## Coverage Gaps

**Untested module:** `src/clef_retrieval/eda.py`
- Function `summarize_lengths()` has no test file
- Risk: Statistical calculations could break unnoticed
- Priority: Medium (utility function, but used in data analysis)

---

*Testing analysis: 2025-01-14*
