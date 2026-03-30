# check-that-clef-2026

## CLEF 2026 Retrieval Pipeline

```bash
uv run python main.py build-index
uv run python main.py predict --lang en --split dev
uv run python main.py evaluate --lang en --split dev
```

- Embeddings: `models/gemini-embedding-2-preview`
- Structured extraction: `gemini-3.1-flash-lite-preview`
- Current reranker stage: lexical overlap fallback (Gemini reranker to be added next)

Operational commands (`build-index`, `predict`, `evaluate`) require `GEMINI_API_KEY` in your environment or `.env`.

```bash
brew install uv
uv sync
uv run main.py --help

# To add or remove dependencies, use the following commands:
uv add <dependency>
uv remove <dependency>

# To run formatting and linting checks, use:
uv run ruff format .
uv run ruff check .

# Authenticate with HuggingFace
uv run hf auth login
```
