# check-that-clef-2026

## Quick start

```bash
brew install uv
uv sync

# Run evaluation pipeline
uv run python -m clef_pipeline.main

# Deploy Modal backend used by the Streamlit demo
uv run modal deploy src/clef_demo/clef_demo/backend.py

# Run Streamlit demo locally (uses the deployed Modal backend)
uv run streamlit run src/clef_demo/clef_demo/streamlit_app.py

# To add or remove dependencies, use the following commands:
uv add <dependency>
uv remove <dependency>

# To run formatting and linting checks, use:
uv run ruff format
uv run ruff check

# Authenticate with HuggingFace
uv run hf auth login
```

## Code Layout

- `src/clef_pipeline/` — packaged retrieval/evaluation system (Modal entrypoint + retrieval stack)
  - `src/clef_pipeline/clef_pipeline/main.py` — Modal app and local entrypoint for multilingual evaluation and optional submission export
  - `src/clef_pipeline/clef_pipeline/pipeline.py` — orchestration layer for indexing, candidate generation, RRF fusion, and reranking
  - `src/clef_pipeline/clef_pipeline/pipeline_config.py` — dataclass config model and preset profiles (`demo`, `evaluation`, `retrieval-only`)
  - `src/clef_pipeline/clef_pipeline/registry.py` — component factories that build retrievers/rerankers and assemble the pipeline
  - `src/clef_pipeline/clef_pipeline/retrievers.py` — dense retrievers (BGE-M3, Harrier) and sparse BM25+ retriever with cache support
  - `src/clef_pipeline/clef_pipeline/rerankers.py` — cross-encoder rerankers (Gemma and Nemotron variants)
  - `src/clef_pipeline/clef_pipeline/metrics.py` — per-language and global aggregation for MRR/Recall metrics
  - `src/clef_pipeline/clef_pipeline/submission.py` — TSV submission file writing and Modal volume download command helpers
  - `src/clef_pipeline/clef_pipeline/interfaces.py` — abstract contracts for retrievers and rerankers
  - `src/clef_pipeline/clef_pipeline/logging_utils.py` — logger setup and simple stage timer utility
  - `src/clef_pipeline/clef_pipeline/utils.py` — constants, ranking helpers, and reciprocal-rank-fusion utility
- `src/clef_demo/` — Streamlit demo package with Modal backend and web app deployment entrypoints
- `src/clef_training/` — training and hard-negative-mining scripts for dense retrieval model development
- `tests/` — regression and modular pipeline tests for retrieval, metrics, configs, and demo integration

## Configuration Profiles

The pipeline uses preset profiles defined in `src/clef_pipeline/clef_pipeline/pipeline_config.py` to specify combinations of retrievers, rerankers, and fusion strategies. You can switch or customize these to experiment with different models.

### Built-in Profiles

- `demo`: Lightweight profile using a small dense retriever (`harrier-270m`), a sparse retriever (`sparse`), and the `nemotron` reranker. Recommended for fast local testing.
- `evaluation`: Heavyweight profile using a large dense retriever (`harrier-27b`), a sparse retriever (`sparse`), and the `nemotron` reranker.
- `retrieval-only`: Uses `harrier-270m` and `sparse`, but completely disables the cross-encoder reranking step.

### Customizing Components

To change components, edit the list of `RetrieverConfig` or `RerankerConfig` within `build_pipeline_config()` in `pipeline_config.py`. 
You can mix and match the available models:
- **Retrievers:** `harrier-270m`, `harrier-27b`, `bge-m3`, `sparse`
- **Rerankers:** `nemotron`, `gemma2b`
- **Fusion Methods:** `rrf` (Reciprocal Rank Fusion), `random_forest` (Learned RF Classifier)

## Modal commands

```bash
# Evaluation pipeline
uv run modal run -m clef_pipeline.main --split dev

# Deploy Streamlit demo on Modal
uv run modal deploy src/clef_demo/clef_demo/modal_app.py

# Streamlit demo on Modal (live-reload dev mode)
uv run modal serve -m clef_demo.modal_app

# Deploy backend used by the Streamlit demo
uv run modal deploy src/clef_demo/clef_demo/backend.py

# Stop deployed demo apps
uv run modal stop-app clef-backend
uv run modal stop-app checkthat-streamlit-demo
```

## Submission Guidelines

https://www.codabench.org/competitions/15611/#/pages-tab

### Prepare files for upload

1. Generate predictions for the competition split:

```bash
uv run modal run -m clef_pipeline.main \
  --split test \
  --export-submission-tsv \
  --submission-volume-subdir submissions \
  --submission-download-dir submissions
```

2. Move into the downloaded run directory (`submissions/test-<timestamp>`) and validate expected files:

```bash
cd submissions/test-<timestamp>
ls predictions_*.tsv
```

3. Create the upload archive:

```bash
zip -r predictions.zip predictions_*.tsv
unzip -l predictions.zip
```

Upload `predictions.zip` to Codabench.

Each team must create only one account in CodaBench and submit their predictions exclusively through that account.
Make sure your account name matches that used during CLEF registration.
The last valid submission will be considered as the final submission!
The submission file predictions.zip must include your predictions.
Inside the predictions.zip file include .tsv files (TSV, not CSV) for language-specific predictions.
The .tsv files must be named predictions_{lang}.tsv where {lang} is either en, de, or fr.
The predictions.zip file must include the prediction .tsv files for the languages you want to participate in.
The predictions_{lang}.tsv must contain the following columns: "index", "preds", where "index" is the index of the post and "preds" contains an array of the top5 predicted pubkeys (in descending order: [pred1, pred2, pred3, pred4, pred5], where pred1 is the pubkey of the highest ranked publication from the collection set)
You are allowed to submit max 50 submissions per day per team.
