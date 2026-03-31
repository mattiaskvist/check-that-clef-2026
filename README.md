# check-that-clef-2026

## CLEF 2026 Retrieval Pipeline

**Performance: 64.7% MRR@5 (EN), 62.6% MRR@5 (FR) — 40% better than baseline**

### Quick Start

```bash
# Build paper index
uv run python main.py build-index

# Generate predictions (RECOMMENDED: skip extraction for best performance)
uv run python main.py predict --lang en --split dev --skip-query-extraction

# Evaluate
uv run python main.py evaluate --lang en --split dev --multilingual-metrics
```

### Pipeline Architecture

- **Embeddings**: `models/gemini-embedding-2-preview` (Google Gemini)
- **Query**: Raw tweet text (structured extraction disabled for optimal performance)
- **Retrieval**: Dense retrieval (top-200 candidates)
- **Reranking**: Jina v2 multilingual cross-encoder (top-50)
- **Disambiguation**: Multi-signal scoring for duplicate titles

### ⚠️ Important: Skip Query Extraction

**Always use `--skip-query-extraction` for best performance.**

Our testing shows:
- **WITH extraction**: 35.6% MRR@5 (EN), 16.3% MRR@5 (FR)
- **WITHOUT extraction**: 64.7% MRR@5 (EN), 62.6% MRR@5 (FR) ✅

Raw tweet text works better than structured extraction. See `.planning/PERFORMANCE-BREAKTHROUGH.md` for details.

Operational commands (`build-index`, `predict`, `evaluate`) require `GEMINI_API_KEY` in your environment or `.env`.

### Development Setup

```bash
# Install uv and dependencies
brew install uv
uv sync

# Authenticate with HuggingFace (for reranker models)
uv run huggingface-cli login

# Set Gemini API key
export GEMINI_API_KEY="your-key-here"
# Or create .env file with: GEMINI_API_KEY=your-key-here
```

### Running Tests

```bash
# Run full test suite
uv run pytest

# Run specific test file
uv run pytest tests/test_pipeline.py -v

# Run with coverage
uv run pytest --cov=src/clef_retrieval
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .
```

### Performance Optimization

Current configuration achieves **64.7% MRR@5** (EN) and **62.6% MRR@5** (FR).

**Key settings:**
- `--skip-query-extraction` (required for optimal performance)
- `rerank_top_k=50` (semantic reranking depth)
- `top_k=200` (dense retrieval candidates)

**Experimental tuning:**
```bash
# Test deeper reranking (may improve further)
# Edit src/clef_retrieval/config.py: rerank_top_k=100

# Test with different reranker backends
# Edit src/clef_retrieval/config.py: reranker_backend="bge_v2_m3"
```

### Project Structure

```
.planning/                  # GSD workflow artifacts
├── PROJECT.md              # Project goals and decisions
├── ROADMAP.md              # Phase breakdown
├── STATE.md                # Current progress
├── PERFORMANCE-BREAKTHROUGH.md  # Analysis of extraction findings
└── phases/
    ├── 01-query-understanding/     # Phase 1 (deprecated - hurts performance)
    ├── 02-semantic-reranking/      # Phase 2 (adds value)
    └── 03-duplicate-disambiguation/ # Phase 3 (adds value)

src/clef_retrieval/
├── config.py               # Pipeline configuration
├── pipeline.py             # Main retrieval logic
├── reranker.py             # Semantic reranking (Jina/BGE)
├── disambiguation.py       # Duplicate title handling
├── gemini_client.py        # Gemini API client
├── paper_index.py          # Paper embedding index
└── schemas.py              # Data models

tests/                      # 115 tests (all passing)
main.py                     # CLI entry point
```

### Phase Summary

**Phase 1: Query Understanding** ❌ DEPRECATED
- Structured extraction hurts performance
- Use raw tweets instead (`--skip-query-extraction`)

**Phase 2: Semantic Reranking** ✅ VALUABLE
- Jina v2 cross-encoder improves ranking quality
- Reranks top-50 dense retrieval candidates

**Phase 3: Duplicate Disambiguation** ✅ VALUABLE
- Handles duplicate-title papers using multi-signal scoring
- Author > Method/Finding > Venue priority

See `.planning/PERFORMANCE-BREAKTHROUGH.md` for detailed analysis.

### Dependencies

- **uv**: Python package manager
- **Python 3.10+**: Required runtime
- **Gemini API**: For embeddings (no extraction needed)
- **HuggingFace**: For semantic reranker models (Jina/BGE)

### Common Commands

```bash
# Full evaluation workflow
uv run python main.py build-index
uv run python main.py predict --lang en --split dev --skip-query-extraction --allow-cost-overrun
uv run python main.py evaluate --lang en --split dev --multilingual-metrics

# Test on subset (faster iteration)
uv run python main.py predict --lang en --split dev --skip-query-extraction \
  --subset-per-language-limit 50 --subset-seed 42
uv run python main.py evaluate --lang en --split dev --multilingual-metrics \
  --subset-per-language-limit 50 --subset-seed 42
```
