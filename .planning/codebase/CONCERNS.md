# Codebase Concerns

**Analysis Date:** 2025-01-10

## Tech Debt

**Experimental Scripts Duplicating Logic:**
- Issue: Core functionality exists in both `geminitry.py` (188 lines) and `OsvaldsTry.py` (223 lines), alongside the clean library in `src/clef_retrieval/`. These scripts contain hardcoded dataset loading, model configurations, and retrieval logic that should be refactored into shared utilities.
- Files: `geminitry.py`, `OsvaldsTry.py`
- Impact: Maintenance burden increases with each bug fix or feature addition. Changes must be replicated across three locations. Makes it unclear which implementation is the source of truth.
- Fix approach: Consolidate experimental code into single `experiments/` directory, or refactor reusable patterns into `src/clef_retrieval/` modules. Keep only one reference implementation.

**Hardcoded Dataset Identifier:**
- Issue: The HuggingFace dataset ID `"sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"` is repeated verbatim in 7 locations across the codebase (`src/clef_retrieval/data.py`, `scorer.py`, `geminitry.py`, `OsvaldsTry.py`, test files).
- Files: `src/clef_retrieval/data.py:8`, `scorer.py:21`, `geminitry.py:25`, `geminitry.py:40`, `OsvaldsTry.py:30`, `OsvaldsTry.py:63`, `tests/test_data_loading.py:21,37`
- Impact: Difficult to update dataset references if the dataset is moved or versioned. Risk of inconsistency if one reference is updated but others aren't.
- Fix approach: Define `DATASET_ID` once as a module constant in `src/clef_retrieval/data.py` (already done) and import it into `scorer.py` and experimental files. Update test mocks to use this constant.

**Magic Numbers and Unmotivated Defaults:**
- Issue: Configuration values like `top_k=200` and `top_n=5` in `src/clef_retrieval/config.py:10-11` lack documentation explaining why these specific values were chosen. No rationale for bounds `(ge=10)` and `(le=20)` on `top_n`.
- Files: `src/clef_retrieval/config.py`
- Impact: Future modifications may select suboptimal values. Difficult to explain design choices to stakeholders.
- Fix approach: Add docstrings explaining the purpose of each config field and reasoning for default/bound values. Document empirical results that informed these choices.

**Unused Main Entry Point:**
- Issue: `main.py:1-7` contains a placeholder "Hello from check-that-clef-2026!" function with no actual implementation. This is the intended CLI entry point but provides no value.
- Files: `main.py`
- Impact: Users cannot run the project via intended entry point. Creates confusion about whether the project is actually runnable.
- Fix approach: Implement actual CLI that orchestrates the retrieval pipeline. Accept language, split, and configuration parameters. Or remove and document that the project is a library only.

## Known Bugs

**Unsafe List Indexing in scorer.py:**
- Symptoms: `scorer.py:31` calls `preds.index(label)` after checking `if label in preds`. If `label` is not an exact match (e.g., due to type mismatch, whitespace, or case sensitivity), it will raise `ValueError`.
- Files: `scorer.py:31`
- Trigger: Run `scorer()` with predictions where labels are strings but dataset returns different types (e.g., integers converted to strings, or vice versa). Also if there are subtle whitespace differences between prediction and label.
- Workaround: Ensure all predictions and labels are normalized to the same type and format before comparison. Add explicit type coercion.
- Fix: Add type validation in `validate_prediction_shape()` to check types. Use try-except around `.index()` call or use safer lookup pattern like `preds.index(label) if label in preds else -1`.

**Potential Type Mismatch in numpy Operations:**
- Symptoms: `src/clef_retrieval/retriever.py:19-20` converts input to numpy array with `np.asarray(scores)` without type specification. If scores contain mixed types or non-numeric values, subsequent operations (`argsort`, `argpartition`) will fail with cryptic numpy errors.
- Files: `src/clef_retrieval/retriever.py:19,28`
- Trigger: Call `topk_indices()` with a list of mixed types like `[0.5, "0.3", None, 0.8]`. Or call with boolean/string array.
- Workaround: Pre-validate input types before calling the function.
- Fix: Add explicit `dtype=float` to `np.asarray()` call. Add runtime validation of input shape and type. Raise descriptive `TypeError` for invalid inputs.

**Silent Failure in normalize_authors():**
- Symptoms: `src/clef_retrieval/data.py:17` returns empty string `""` for `None` input, but also returns `""` for empty list. Callers cannot distinguish between "author data not provided" and "author list was empty".
- Files: `src/clef_retrieval/data.py:11-17`
- Trigger: Call `normalize_authors(None)` and `normalize_authors([])` - both return `""`.
- Workaround: Check source of None before calling function.
- Fix: Return `None` instead of empty string for unprovided data. Use Optional[str] type hint. Or raise explicit exception for None input.

## Security Considerations

**API Key Exposure via Environment Variables:**
- Risk: Both `geminitry.py:18` and configuration expect `GEMINI_API_KEY` environment variable. If this is logged, committed to git, or appears in error messages, credentials leak.
- Files: `geminitry.py:18`, `.env` file (not read per policy)
- Current mitigation: Code uses `load_dotenv()` from `python-dotenv` which isolates .env from code. No explicit logging of API keys observed in source.
- Recommendations: 
  - Add `.env` and `.env.local` to `.gitignore` (verify they're not tracked).
  - Review all error handling to ensure no API keys are printed in exceptions.
  - Consider using OAuth or service account keys instead of API key strings.
  - Add secret scanning to CI/CD if not already present.

**Dataset Access Not Validated:**
- Risk: `src/clef_retrieval/data.py:20-25` loads datasets from HuggingFace without verifying dataset integrity or authenticity. If the dataset source is compromised, malicious data could be loaded.
- Files: `src/clef_retrieval/data.py:20-25`, `scorer.py:21`, experimental files
- Current mitigation: Uses official HuggingFace library which has checksum validation. But no explicit verification in application code.
- Recommendations:
  - Document that datasets must be loaded only from the official `sschellhammer/` namespace.
  - Consider pinning dataset version/commit hash in code.
  - Validate dataset schema on load (compare column names and types to expected schema).

**Language and Split Validation Incomplete:**
- Risk: `scorer.py:18-19` validates language and split with assertions, but assertions can be disabled with `python -O`. If someone runs code with optimization flags, validation is bypassed.
- Files: `scorer.py:18-19`
- Current mitigation: Assertions are present during normal execution.
- Recommendations:
  - Replace assertions with explicit if-checks that raise `ValueError` or `ValueError`. This cannot be disabled.
  - Example: `if lang not in ["de", "en", "fr"]: raise ValueError(f"Invalid language: {lang}")`

## Performance Bottlenecks

**Dense Embedding Computation Without Caching:**
- Problem: `geminitry.py:40-80` and `OsvaldsTry.py:63-100` generate embeddings for every title in the collection on each script run, without caching results. With potentially thousands of papers, this is redundant and slow.
- Files: `geminitry.py:64-80`, `OsvaldsTry.py:91-100`
- Cause: Embeddings are computed fresh each run. No disk or memory cache is used.
- Improvement path:
  - Implement embedding cache in `src/clef_retrieval/` module (e.g., SQLite or pickle store).
  - Load cached embeddings if available before computing new ones.
  - Invalidate cache when collection dataset version changes.
  - Use `src/clef_retrieval/config.py:12` cache_dir to store embeddings.

**Unbounded Similarity Computation:**
- Problem: Retrieval pipeline must compute similarity scores between query embedding and all collection embeddings. With large collections (potentially 100k+ papers), this requires O(n) operations per query.
- Files: `src/clef_retrieval/retriever.py:18-29`
- Cause: No pre-filtering or index structures (e.g., FAISS, vector databases). Using raw numpy operations.
- Improvement path:
  - Integrate FAISS or Annoy for approximate nearest neighbor search.
  - Build index on collection embeddings once, then do sub-linear queries.
  - Trade-off: Small accuracy loss for 10x+ speed improvement on large collections.

**50+ Print Statements for Debugging:**
- Problem: `geminitry.py` and `OsvaldsTry.py` contain ~50 print statements, suggesting heavy reliance on print debugging instead of structured logging. This impacts performance on production runs and clutters output.
- Files: `geminitry.py`, `OsvaldsTry.py`
- Cause: Scripts written for interactive experimentation, not production use.
- Improvement path: 
  - Replace print() with Python logging module with configurable log levels.
  - Remove or set to debug level to reduce output volume.
  - This allows end users to control verbosity without modifying code.

## Fragile Areas

**Brittle Type Handling in Schemas:**
- Files: `src/clef_retrieval/schemas.py`
- Why fragile: 
  - `TweetEvidence.language` defaults to `"unknown"` string (line 8) but no validation that it matches expected ISO codes.
  - `PaperEvidence.venue` defaults to empty string (line 25) but can be silently missing.
  - List fields use `Field(default_factory=list)` which is correct, but no validation that lists contain appropriate types (e.g., list[str]).
- Safe modification: 
  - Add Pydantic validators using `@field_validator` to enforce ISO 639-1 language codes.
  - Use `Optional[str]` for potentially missing fields rather than empty string defaults.
  - Add validation for list element types.
  - Add docstrings explaining expected values for each field.
- Test coverage: `tests/test_schemas.py` only tests basic instantiation; no validation testing.

**Fragile EDA Summary Function:**
- Files: `src/clef_retrieval/eda.py:6-10`
- Why fragile:
  - `summarize_lengths()` assumes input is list of strings. If None values are passed, it silently converts to "None" string.
  - No handling for non-iterable values or malformed text.
  - Uses `statistics.mean()` and `median()` which raise `StatisticsError` on empty list.
- Safe modification:
  - Add type validation and early return for empty/None inputs (already handles empty with return line 9).
  - Add explicit None filtering: `lengths = [len((value or "").split()) for value in values if value]`
  - Document that None values are treated as empty strings.
- Test coverage: No test file exists for `eda.py`.

**Experiment Scripts Lack Error Recovery:**
- Files: `geminitry.py:64-80`, `OsvaldsTry.py:91-100`
- Why fragile:
  - Both scripts use `try/except Exception` handlers that catch all exceptions equally (generic exception handling).
  - In `geminitry.py:80`, the exception is caught and printed but execution continues with undefined state.
  - No logging of which embedding failed or how many succeeded.
  - No checkpoint/resume capability if script crashes midway through large collection.
- Safe modification:
  - Catch specific exceptions (e.g., `RequestError`, `APIError`) rather than bare `Exception`.
  - Log failures with context (which paper, which model, timestamp).
  - Implement checkpoint system to resume from last successful embedding.
  - Add CLI flag to limit number of embeddings to process for testing.

## Scaling Limits

**Single-Machine Processing:**
- Current capacity: All scripts load full dataset into memory and process sequentially. No distributed processing.
- Limit: Will hit memory limits with very large collections (1M+ papers). Processing time is linear in collection size with no parallelization.
- Scaling path:
  - Implement batch processing with memory-efficient streaming.
  - Use `datasets.IterableDataset` instead of regular Dataset for large collections.
  - Add multiprocessing for embedding computation using `torch.DataLoader` or similar.
  - Consider distributed compute frameworks (Ray, Spark) for production-scale retrieval.

**No Persistence of Intermediate Results:**
- Current capacity: Each run recomputes embeddings and retrieval results from scratch. No saved checkpoints.
- Limit: Useful for experimentation but prevents iteration. Can't easily compare different ranking algorithms against same embeddings.
- Scaling path:
  - Store embeddings in a vector database (Pinecone, Weaviate, Qdrant).
  - Cache query results with TTL for repeated queries.
  - Use cache_dir from config (`src/clef_retrieval/config.py:12`) to persist computed embeddings.

## Dependencies at Risk

**Python Version Requirement is Unrealistic:**
- Risk: `pyproject.toml:6` specifies `requires-python = ">=3.14"`. Python 3.14 does not exist yet (as of 2025). This will prevent the project from being installed on any current system.
- Impact: pip/uv will fail to find compatible environment. Project cannot be used.
- Migration plan: Change to `requires-python = ">=3.10"` or a realistic version based on actual dependencies. Pin major.minor version, not micro (3.10, not 3.14).

**Heavy Dependency on External APIs:**
- Risk: Core retrieval pipeline depends on Google Gemini API (in config) and HuggingFace Datasets service. No offline fallback.
- Impact: Network outages or API rate limiting will block the entire system. API deprecation or pricing changes directly impact application.
- Migration plan:
  - Implement abstraction for embedding models to support multiple providers (Gemini, Ollama, sentence-transformers).
  - Cache embeddings locally to reduce API calls.
  - Implement offline mode using local models (ollama is already included as dependency).
  - Document fallback procedures if APIs are unavailable.

**Incompatible Optional Dependencies:**
- Risk: `ollama` package is in dependencies but `OsvaldsTry.py` requires it, yet `geminitry.py` requires `google-genai`. Both are always installed but only one is used per script.
- Impact: Unnecessary bloat. Requires users to install heavy ML libraries even if they only need one path.
- Migration plan:
  - Create optional dependency groups in `pyproject.toml`: `dev`, `gemini`, `ollama`.
  - Move `ollama` and `google-genai` to optional groups.
  - Document which group is needed for which use case.
  - Example: `uv sync --extras gemini` or `uv sync --extras ollama`.

**Torch and Transformers Included but Not Used:**
- Risk: `torch` (235+ MB) and `transformers` are installed but appear unused in core library code.
- Impact: Large download size. Suggests incomplete migration away from these dependencies or incomplete implementation.
- Migration plan:
  - Audit code to determine if torch/transformers are actually used.
  - If only needed by experimental scripts, move to optional `dev` group.
  - If being prepared for future features, document the plan clearly.
  - Remove from core dependencies if truly unused.

## Missing Critical Features

**No Evaluation on Full Test Set:**
- Problem: `scorer.py` accepts a split parameter ("train" or "dev") but it's unclear if evaluation is performed on the full test set for final submissions. No command or documented procedure to evaluate on all data.
- Blocks: Cannot compute official competition scores for submission.
- Fix: Implement proper evaluation workflow:
  - Add explicit "test" split support if available in dataset.
  - Document how to generate submission file.
  - Add validation that exactly N predictions are made for N queries.

**No Model Persistence or Serialization:**
- Problem: Trained embeddings and ranked results are computed fresh each run. No way to save or reload a model for inference.
- Blocks: Cannot deploy system to production. Cannot batch-generate results and store for later use.
- Fix: Implement model export:
  - Save embeddings to disk/database in production format.
  - Implement export to standard formats (ONNX for embeddings, CSV for rankings).
  - Add versioning to track which embeddings/rankings correspond to which model/dataset version.

**No Validation on Prediction Format:**
- Problem: `src/clef_retrieval/evaluation.py:6-7` only validates that each row has exactly 5 predictions. Does not validate:
  - That predictions are actual paper IDs (pubkeys).
  - That there are no duplicates within a row.
  - That all predicted pubkeys exist in collection.
- Blocks: Invalid predictions will silently get scored as 0, without alerting user to format errors.
- Fix: Extend validation function to check:
  - Predictions are strings/pubkeys that exist in collection.
  - No duplicates per row.
  - No missing predictions.
  - Raise descriptive error on validation failure.

## Test Coverage Gaps

**No Tests for EDA Module:**
- What's not tested: `src/clef_retrieval/eda.py` has no test file. `summarize_lengths()` is not tested for edge cases (None values, empty lists, very long strings).
- Files: `src/clef_retrieval/eda.py`
- Risk: Changes to this function could break hidden callers without warning. Edge cases like None, empty list, malformed input are untested.
- Priority: Medium

**No Integration Tests:**
- What's not tested: End-to-end pipeline from raw tweet to ranked papers. Current tests are unit tests of individual functions. No test that verifies the full retrieval system works together.
- Files: All `tests/test_*.py` are unit tests only. No `test_integration.py` or `test_e2e.py`.
- Risk: Individual components work in isolation but integration bugs could exist. For example, embedding dimensions mismatch, or schema field names not matching retrieval code.
- Priority: High

**No Tests for Experimental Scripts:**
- What's not tested: `geminitry.py`, `OsvaldsTry.py`, `main.py`, `scorer.py` have zero test coverage.
- Files: `geminitry.py`, `OsvaldsTry.py`, `main.py`, `scorer.py`
- Risk: These are user-facing scripts. Bug in `scorer.py` directly impacts evaluation results. Cannot refactor without breaking changes.
- Priority: High - scorer.py especially needs tests because it computes official metrics.

**No Tests for Error Cases:**
- What's not tested: What happens when:
  - Dataset cannot be loaded (network error, authentication failure).
  - API call fails (rate limit, invalid key).
  - Invalid data types are passed to functions.
  - Empty/None inputs to validation functions.
- Files: All test files
- Risk: Error handling code is never executed in tests, so bugs in exception paths remain hidden.
- Priority: Medium - at least smoke tests for common failure modes.

**No Test for Type Validation:**
- What's not tested: Pydantic models in `src/clef_retrieval/schemas.py` are validated on instantiation, but no test verifies validation actually rejects invalid inputs.
- Files: `tests/test_schemas.py:12-23` only tests valid instantiation. No tests for rejected inputs.
- Risk: Schema validation could silently fail if Pydantic version changes or if type hints become incorrect.
- Priority: Low - Pydantic is well-tested library, but defensive programming is good practice.

---

*Concerns audit: 2025-01-10*
