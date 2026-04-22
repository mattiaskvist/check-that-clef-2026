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
- **Rerankers:** `nemotron`, `gemma2b`, `jina-v3`
- **Fusion Methods:** `rrf` (Reciprocal Rank Fusion), `random_forest` (Learned RF Classifier)

## Ablation Study

Run the following commands to execute the ablation study and collect the metrics/submission files for each combination. The metrics are saved to local JSON files (`metrics_X.json`) and submission files will be generated in the Modal volume and instructions to download them will be printed.

### Recommended Execution Order (for maximum caching)

The Random Forest fuser trains automatically the first time it is needed and saves to the Modal volume. To avoid recomputing dense document and query embeddings multiple times, run the commands in this order:

1. **Run Command 5 (Hybrid RRF)**: Builds the massive `harrier-27b` document embeddings, the `harrier-27b` dev queries, and the optimized `sparse` dev queries.
2. **Run Command 6 (Hybrid RF)**: Hits the document cache from Command 5. Automatically computes the `train` queries for `harrier-27b` and `sparse`, trains the Random Forest model, saves it to the cache, and evaluates.
3. **Run Commands 7, 8, 9, 10, 11, 12, 13, 2, and 4**: These will now completely hit the caches built in steps 1 and 2, running lightning fast and only spending time on reranking where applicable.
4. **Run Command 1 (Vanilla BM25)**: Computes a new sparse cache for the `dev` queries using vanilla BM25 parameters.
5. **Run Command 3 (BGE-M3)**: Computes the `bge-m3` document and dev query embeddings from scratch.

*(Note: If you ever need to explicitly force the random forest model to retrain, simply append `--force-retrain-fusion` to the relevant command).*

### 1. Vanilla BM25 Sparse Retriever
```bash
uv run modal run -m clef_pipeline.main \
  --profile custom --dense-model "" --disable-reranker --sparse-vanilla \
  --split dev --export-submission-tsv --metrics-output-file metrics_1_vanilla_bm25.json
```

### 2. Optimized Sparse Retriever
```bash
uv run modal run -m clef_pipeline.main \
  --profile custom --dense-model "" --disable-reranker \
  --split dev --export-submission-tsv --metrics-output-file metrics_2_optimized_sparse.json
```

### 3. BGE-M3 Dense Retriever Only
```bash
uv run modal run -m clef_pipeline.main \
  --profile custom --dense-model "bge-m3" --disable-sparse --disable-reranker \
  --split dev --export-submission-tsv --metrics-output-file metrics_3_bgem3_dense.json
```

### 4. Harrier 27B Dense Retriever Only
```bash
uv run modal run -m clef_pipeline.main \
  --profile custom --dense-model "harrier-27b" --disable-sparse --disable-reranker \
  --split dev --export-submission-tsv --metrics-output-file metrics_4_harrier_dense.json
```

### 5. Hybrid (Optimized Sparse + Harrier) + RRF Fusion
```bash
uv run modal run -m clef_pipeline.main \
  --profile custom --dense-model "harrier-27b" --fusion-method "rrf" --disable-reranker \
  --split dev --export-submission-tsv --metrics-output-file metrics_5_hybrid_rrf.json
```

### 6. Hybrid (Optimized Sparse + Harrier) + Random Forest Fusion
```bash
uv run modal run -m clef_pipeline.main \
  --profile custom --dense-model "harrier-27b" --fusion-method "random_forest" --disable-reranker \
  --split dev --export-submission-tsv --metrics-output-file metrics_6_hybrid_rf.json
```

### 7. Hybrid + Random Forest Fusion + Nemotron Reranker
```bash
uv run modal run -m clef_pipeline.main \
  --profile custom --dense-model "harrier-27b" --fusion-method "random_forest" \
  --split dev --export-submission-tsv --metrics-output-file metrics_7_hybrid_rf_nemotron.json
```

### 8. Hybrid + RRF Fusion + Nemotron Reranker
```bash
uv run modal run -m clef_pipeline.main \
  --profile custom --dense-model "harrier-27b" --fusion-method "rrf" \
  --split dev --export-submission-tsv --metrics-output-file metrics_8_hybrid_rrf_nemotron.json
```

### 9. Harrier 27B Dense Only + Nemotron Reranker
```bash
uv run modal run -m clef_pipeline.main \
  --profile custom --dense-model "harrier-27b" --disable-sparse \
  --split dev --export-submission-tsv --metrics-output-file metrics_9_harrier_nemotron.json
```

### 10. Hybrid + Random Forest Fusion + Gemma Reranker
```bash
uv run modal run -m clef_pipeline.main \
  --profile custom --dense-model "harrier-27b" --fusion-method "random_forest" --reranker-model "gemma2b" \
  --split dev --export-submission-tsv --metrics-output-file metrics_10_hybrid_rf_gemma.json
```

### 11. Hybrid + RRF Fusion + Gemma Reranker
```bash
uv run modal run -m clef_pipeline.main \
  --profile custom --dense-model "harrier-27b" --fusion-method "rrf" --reranker-model "gemma2b" \
  --split dev --export-submission-tsv --metrics-output-file metrics_11_hybrid_rrf_gemma.json
```

### 12. Hybrid + Random Forest Fusion + Jina Reranker
```bash
uv run modal run -m clef_pipeline.main \
  --profile custom --dense-model "harrier-27b" --fusion-method "random_forest" --reranker-model "jina-v3" \
  --split dev --export-submission-tsv --metrics-output-file metrics_12_hybrid_rf_jina.json
```

### 13. Hybrid + RRF Fusion + Jina Reranker
```bash
uv run modal run -m clef_pipeline.main \
  --profile custom --dense-model "harrier-27b" --fusion-method "rrf" --reranker-model "jina-v3" \
  --split dev --export-submission-tsv --metrics-output-file metrics_13_hybrid_rrf_jina.json
```

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
