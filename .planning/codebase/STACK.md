# Technology Stack

**Analysis Date:** 2025-01-03

## Languages

**Primary:**
- Python 3.14+ - All application code, ML models, and data processing

## Runtime

**Environment:**
- Python 3.14 or higher (as specified in `pyproject.toml`)

**Package Manager:**
- UV (modern Python package manager)
- Lockfile: `uv.lock` present and maintained

## Frameworks

**Core:**
- Pydantic 2.12.5+ - Data validation and schema definition (`src/clef_retrieval/schemas.py`, `src/clef_retrieval/config.py`)
- Datasets 4.8.4+ - HuggingFace dataset loading and management (`src/clef_retrieval/data.py`)

**ML/AI:**
- PyTorch 2.11.0+ - Deep learning framework for model inference
- Transformers 5.3.0+ - HuggingFace transformers library for NLP models
- Ollama 0.6.1+ - Local LLM inference engine (`OsvaldsTry.py` uses `ollama.pull()` and embedding generation)
- Google GenAI 1.68.0+ - Google Gemini API integration (`geminitry.py` uses `google.genai`)

**Testing:**
- Pytest 8.4.2+ - Test framework and runner
  - Config: `pyproject.toml` includes `[tool.pytest.ini_options]` with pythonpath configuration

**Linting/Formatting:**
- Ruff 0.15.7+ - Fast Python linter and formatter
  - Commands: `uv run ruff format` and `uv run ruff check`

**Utilities:**
- Python-dotenv 1.2.2 - Environment variable loading (`geminitry.py` uses `load_dotenv()`)
- NumPy 2.4.3+ - Numerical computing (dense retrieval operations in `src/clef_retrieval/retriever.py`)
- IPykernel 7.2.0+ - Jupyter notebook kernel support (for `CT26_Task1_baseline.ipynb`)

## Key Dependencies

**Critical:**
- `datasets>=4.8.4` - Core dependency for loading CT26 dataset from HuggingFace Hub
  - Dataset ID: `"sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"`
  - Used to load collection data, language-specific splits (de, en, fr) in both training and dev modes
  - Imported in: `src/clef_retrieval/data.py`, `geminitry.py`, `OsvaldsTry.py`, `scorer.py`

- `google-genai>=1.68.0` - Google Gemini AI API client
  - Used for zero-shot paper extraction and reranking
  - Supports structured output with Pydantic schema validation
  - JSON mode and temperature control for deterministic extraction
  - Imported in: `geminitry.py` (`from google import genai`)

- `ollama>=0.6.1` - Local LLM inference (Ollama client)
  - Pulls and runs local models like `llama3.1`
  - Embedding generation for dense retrieval
  - Cache management for embeddings
  - Imported in: `OsvaldsTry.py`

- `pydantic>=2.12.5` - Data validation for configuration and schemas
  - `RetrievalConfig` in `src/clef_retrieval/config.py`
  - `TweetEvidence` and `PaperEvidence` in `src/clef_retrieval/schemas.py`
  - Structured schema enforcement in API responses

**Infrastructure:**
- `huggingface-hub>=1.7.2` - HuggingFace Hub API (dependency of `datasets`)
  - Enables authenticated access to HuggingFace datasets
  - Authentication via `hf auth login` (mentioned in README.md)
- `torch>=2.11.0` - PyTorch for model inference
- `transformers>=5.3.0` - HuggingFace model library for NLP tasks

## Configuration

**Environment:**
- Configuration via `src/clef_retrieval/config.py` (`RetrievalConfig` Pydantic model)
  - `embedding_model` - Default: `"models/gemini-embedding-2-preview"` (Google Gemini embedding model)
  - `rerank_model` - Default: `"gemini-3.1-flash-lite-preview"` (Google Gemini reranking model)
  - `query_model` - Default: `"gemini-3.1-flash-lite-preview"` (Google Gemini query model)
  - `top_k` - Default: 200, range [10, ∞]
  - `top_n` - Default: 5, range [1, 20]
  - `cache_dir` - Default: `".cache/clef_retrieval"`

- Environment variables (via `python-dotenv`):
  - `GEMINI_API_KEY` - Google Gemini API key (automatically loaded by `google.genai.Client()`)
  - `HF_TOKEN` - HuggingFace authentication token for hub access (optional, used with `hf auth login`)

**Build:**
- `pyproject.toml` - Python project configuration (PEP 518/PEP 621 standard)
  - Declares all dependencies and versions
  - Pytest configuration with pythonpath
  - Development dependencies group (pytest, ruff)

## Platform Requirements

**Development:**
- macOS/Linux/Windows with Python 3.14+
- `brew install uv` (for macOS) or equivalent for other platforms
- HuggingFace authentication: `uv run hf auth login`
- Google Gemini API key (stored in `.env` via `python-dotenv`)
- Optional: Ollama installed locally for local LLM inference

**Production:**
- Python 3.14+ runtime
- Network access to HuggingFace Hub API
- Network access to Google Gemini API (for `geminitry.py` usage)
- Optional: Local Ollama installation (for `OsvaldsTry.py` usage)
- Cache directory writeable (default: `.cache/clef_retrieval/`)

---

*Stack analysis: 2025-01-03*
