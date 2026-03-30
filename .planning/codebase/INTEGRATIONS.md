# External Integrations

**Analysis Date:** 2025-01-03

## APIs & External Services

**Language Models (LLM):**
- Google Gemini API (geminitry.py)
  - SDK/Client: `google-genai>=1.68.0`
  - Auth: `GEMINI_API_KEY` environment variable
  - Models used:
    - `gemini-2.5-flash` - Cost-effective generation in `geminitry.py`
    - `gemini-3.1-flash-lite-preview` - Reranking and query processing in `src/clef_retrieval/config.py`
  - Capabilities: Structured output with Pydantic schema validation, JSON mode, zero-shot extraction
  - Usage: Paper title/author/keyterm extraction from tweets, reranking candidate documents

- Ollama Local LLM Server (OsvaldsTry.py)
  - SDK/Client: `ollama>=0.6.1`
  - Auth: None (runs locally)
  - Models used:
    - `llama3.1` - Locally pulled and executed for dense retrieval
  - Capabilities: Local embedding generation, model management
  - Usage: Dense retrieval with local models, embedding cache management

**Data APIs:**
- HuggingFace Hub (Datasets API)
  - SDK/Client: `huggingface-hub>=1.7.2`, `datasets>=4.8.4`
  - Auth: `HF_TOKEN` (optional, can use `hf auth login` for authenticated access)
  - Dataset: `sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims`
  - Data loaded:
    - Collection split: Scientific paper metadata (title, pubkey, abstract, authors)
    - Language-specific splits: `de`, `en`, `fr` (dev and train splits)
    - Tweet claims with evidence about scientific papers
  - Usage: Primary data source for both model input and evaluation labels
  - Integration points: `src/clef_retrieval/data.py`, `geminitry.py`, `OsvaldsTry.py`, `scorer.py`

## Data Storage

**Databases:**
- None detected - Project uses in-memory data from HuggingFace datasets

**File Storage:**
- Local filesystem only
  - Cache directory: `.cache/clef_retrieval/` (default, configurable in `RetrievalConfig`)
  - Embeddings cache: `collection_embeddings.npy` (NumPy binary format in `OsvaldsTry.py`)
  - Notebook: `CT26_Task1_baseline.ipynb` (Jupyter notebook)

**Caching:**
- NumPy embedding cache: `OsvaldsTry.py` saves/loads embeddings to avoid regeneration
- HuggingFace datasets cache: Automatic through `datasets` library
- Local model cache: Ollama manages model storage locally

## Authentication & Identity

**Auth Provider:**
- Google Cloud (Gemini API)
  - Implementation: API key-based authentication
  - Key env var: `GEMINI_API_KEY` (auto-loaded by `google.genai.Client()`)
  - No manual token management required; SDK handles key discovery

- HuggingFace Hub
  - Implementation: Token-based authentication (optional for public datasets)
  - Auth method: `hf auth login` CLI command (documented in `README.md`)
  - Env var: `HF_TOKEN` (optional, managed by huggingface_hub)
  - Usage: Authenticated downloads of the CT26 dataset from private/restricted access if needed

- Ollama
  - Implementation: Local authentication (no external auth required)
  - Runs as local service, manages model files

## Monitoring & Observability

**Error Tracking:**
- None detected - No external error tracking service configured

**Logs:**
- Print-based logging only
  - `geminitry.py`: Print statements for model availability, dataset loading, extraction results
  - `OsvaldsTry.py`: Print statements for model pulling, embedding generation progress
  - Verbose output via `tqdm` progress bars in embedding generation
  - No centralized logging framework

**Debugging:**
- Exception handling: Try-catch in `geminitry.py` for LLM extraction errors with fallback returns
- Manual error inspection via print output

## CI/CD & Deployment

**Hosting:**
- Not detected - Project appears to be development/research oriented
- No deployment configuration found

**CI Pipeline:**
- Not detected - No GitHub Actions or other CI/CD workflows configured
- `.github/` directory exists but contains GSD tools, not CI workflows

**Local Development:**
- UV for reproducible environments: `uv sync`
- Testing: `pytest` with pythonpath configuration for src imports
- Linting: `ruff format` and `ruff check`

## Environment Configuration

**Required env vars:**
- `GEMINI_API_KEY` - Google Gemini API key (CRITICAL for `geminitry.py`)
- `HF_TOKEN` - HuggingFace authentication (optional, only needed for private dataset access)

**Optional configuration:**
- `.cache/clef_retrieval/` - Configurable cache directory via `RetrievalConfig.cache_dir`
- Model selections in `RetrievalConfig` (embedding, rerank, query models)
- `top_k` and `top_n` parameters for retrieval

**Secrets location:**
- `.env` file (loaded via `python-dotenv` in `geminitry.py`)
- HuggingFace CLI credentials in `~/.huggingface/token` (after `hf auth login`)
- No secrets committed to repository (`.env` should be in `.gitignore`)

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected - Project performs batch evaluation, no webhook integrations

## Dataset Configuration

**CT26 Task 1 Dataset:**
- Source: HuggingFace Hub (`sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims`)
- Splits:
  - `collection` - Scientific paper metadata (title, pubkey, abstract, authors, etc.)
  - `en`, `de`, `fr` - Language-specific claim splits (English, German, French)
  - Each language has `train` and `dev` splits
- Evaluation metric: MRR@5 (Mean Reciprocal Rank at top-5)
  - Computed in `scorer.py` via `scorer(top5_preds, lang, split)` function
  - Labels are paper `pubkey` (publication key) values

## Model Configuration

**Embedding Models:**
- `models/gemini-embedding-2-preview` (default in `RetrievalConfig`)
  - Provider: Google Gemini
  - Usage: Dense retrieval query and document embeddings

**Reranking Models:**
- `gemini-3.1-flash-lite-preview` (default in `RetrievalConfig`)
  - Provider: Google Gemini
  - Usage: Reranking candidate papers returned from dense retrieval

**Query Processing Models:**
- `gemini-3.1-flash-lite-preview` (default in `RetrievalConfig`)
  - Provider: Google Gemini
  - Usage: Structured extraction of query intent from tweets

---

*Integration audit: 2025-01-03*
