# CLEF Pipeline

`src/clef_pipeline` contains the retrieval, fusion, reranking, evaluation, and
submission-export pipeline. The primary entrypoint is the Modal app exposed by
`clef_pipeline.main`.

## What It Does

For each query language (`de`, `fr`, `en`), the pipeline:

- loads the CheckThat collection and query split from Hugging Face;
- indexes the collection with enabled dense and sparse retrievers;
- precomputes per-language query caches;
- fuses retriever candidates with RRF or Random Forest fusion;
- optionally reranks fused candidates with a cross-encoder or generative
  reranker;
- reports MRR/Recall metrics for labeled splits; and
- optionally writes Codabench-compatible TSV files.

The expensive parts run in Modal on GPU-backed containers. Dense embeddings,
sparse rankings, fusion models, and submission artifacts are persisted in the
Modal volume `checkthat-embedding-cache`.

## Components

Configured retrievers are resolved through `clef_pipeline.registry`:

- `sparse`: BM25-style retriever with optional bigrams and translation.
- `harrier-270m`: Microsoft Harrier 270M dense retriever.
- `harrier-27b`: Microsoft Harrier 27B dense retriever.
- `bge-m3`: BGE-M3 with the project LoRA adapter
  `boyes-boys-clef-2026/bge-m3-checkthat-finetuned`.

Configured rerankers are:

- `nemotron`
- `gemma2b`
- `jina-v3`
- `qwen3-reranker-8b`
- `qwen3-reranker-4b`
- `qwen3-reranker-0.6b`

Fusion methods are:

- `rrf`: static reciprocal rank fusion.
- `random_forest`: learned candidate fusion using dense/sparse score and rank
  features.

## Prerequisites

From the repository root:

```bash
uv sync
uv run modal setup
uv run modal secret create hf-token HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

The selected models must be accessible to the Hugging Face token stored in the
Modal secret named `hf-token`.

## Profiles And CLI Options

Built-in profiles are defined in `clef_pipeline.pipeline_config`:

- `demo`: `harrier-270m` plus `sparse`, with `nemotron` reranking.
- `evaluation`: `harrier-27b` plus `sparse`, with `nemotron` reranking.
- `retrieval-only`: `harrier-270m` plus `sparse`, no reranker.
- `custom`: configured from CLI flags such as `--dense-model`,
  `--disable-sparse`, `--reranker-model`, and `--disable-reranker`.

Default local entrypoint options include:

```bash
uv run modal run -m clef_pipeline.main \
  --split dev \
  --profile custom \
  --dense-model harrier-27b \
  --fusion-method rrf \
  --reranker-model nemotron
```

Valid splits are `train`, `dev`, and `test`. The `test` split has no labels, so
it supports prediction export but not metrics export.

## Evaluation Runs

Run the default evaluation profile:

```bash
uv run modal run -d -m clef_pipeline.main
```

Run dense-only BGE-M3 retrieval:

```bash
uv run modal run -d -m clef_pipeline.main \
  --profile custom \
  --dense-model bge-m3 \
  --disable-sparse \
  --disable-reranker \
  --split dev
```

Run sparse-only retrieval:

```bash
uv run modal run -d -m clef_pipeline.main \
  --profile custom \
  --dense-model "" \
  --disable-reranker \
  --split dev
```

Run hybrid retrieval with learned fusion and no reranker:

```bash
uv run modal run -d -m clef_pipeline.main \
  --profile custom \
  --dense-model harrier-27b \
  --fusion-method random_forest \
  --disable-reranker \
  --split dev
```

Write metrics to a local JSON file for labeled splits:

```bash
uv run modal run -d -m clef_pipeline.main \
  --split dev \
  --metrics-output-file metrics_dev.json
```

## Cache Controls

The pipeline fingerprints source text and model settings so cached artifacts can
be reused safely between runs.

Force sparse query cache rebuild:

```bash
uv run modal run -d -m clef_pipeline.main --force-recompute-sparse-cache
```

Force dense document embedding rebuild:

```bash
uv run modal run -d -m clef_pipeline.main --force-recompute-dense-documents
```

Force dense query embedding rebuild:

```bash
uv run modal run -d -m clef_pipeline.main --force-recompute-dense-queries
```

Force Random Forest fusion retraining:

```bash
uv run modal run -d -m clef_pipeline.main \
  --fusion-method random_forest \
  --force-retrain-fusion
```

## Ablation Commands

The following commands are useful for comparing retrieval and reranking stages.
Run cache-building configurations first to avoid repeated dense embedding work.

Vanilla BM25:

```bash
uv run modal run -d -m clef_pipeline.main \
  --profile custom --dense-model "" --disable-reranker --sparse-vanilla \
  --split dev --export-submission-tsv \
  --metrics-output-file metrics_1_vanilla_bm25.json
```

Optimized sparse:

```bash
uv run modal run -d -m clef_pipeline.main \
  --profile custom --dense-model "" --disable-reranker \
  --split dev --export-submission-tsv \
  --metrics-output-file metrics_2_optimized_sparse.json
```

BGE-M3 dense only:

```bash
uv run modal run -d -m clef_pipeline.main \
  --profile custom --dense-model bge-m3 --disable-sparse --disable-reranker \
  --split dev --export-submission-tsv \
  --metrics-output-file metrics_3_bgem3_dense.json
```

Harrier 27B dense only:

```bash
uv run modal run -d -m clef_pipeline.main \
  --profile custom --dense-model harrier-27b --disable-sparse --disable-reranker \
  --split dev --export-submission-tsv \
  --metrics-output-file metrics_4_harrier_dense.json
```

Hybrid RRF:

```bash
uv run modal run -d -m clef_pipeline.main \
  --profile custom --dense-model harrier-27b --fusion-method rrf \
  --disable-reranker --split dev --export-submission-tsv \
  --metrics-output-file metrics_5_hybrid_rrf.json
```

Hybrid Random Forest:

```bash
uv run modal run -d -m clef_pipeline.main \
  --profile custom --dense-model harrier-27b --fusion-method random_forest \
  --disable-reranker --split dev --export-submission-tsv \
  --metrics-output-file metrics_6_hybrid_rf.json
```

Hybrid with reranking:

```bash
uv run modal run -d -m clef_pipeline.main \
  --profile custom --dense-model harrier-27b --fusion-method random_forest \
  --reranker-model nemotron \
  --split dev --export-submission-tsv \
  --metrics-output-file metrics_7_hybrid_rf_nemotron.json
```

Swap `--fusion-method rrf` or `--reranker-model gemma2b`, `jina-v3`, or a Qwen
registry name to run additional combinations.

## Submission Export

Generate final prediction files for the unlabeled competition split:

```bash
uv run modal run -m clef_pipeline.main \
  --split test \
  --export-submission-tsv \
  --submission-volume-subdir submissions \
  --submission-download-dir submissions
```

At the end of the run, the CLI prints a `modal volume get` command similar to:

```bash
uv run modal volume get checkthat-embedding-cache /submissions/test-20260416-191700 submissions
```

Create the Codabench upload archive from the downloaded run directory:

```bash
cd submissions/test-<timestamp>
ls predictions_*.tsv
zip -r predictions.zip predictions_*.tsv
unzip -l predictions.zip
```

Expected files are:

- `predictions_de.tsv`
- `predictions_en.tsv`
- `predictions_fr.tsv`

Each TSV contains `index` and `preds`, where `preds` is a top-5 list of
publication keys in descending rank order.

## Resource Expectations

- `harrier-27b` and the larger rerankers require substantial GPU memory.
- The default Modal evaluation function currently requests an H100.
- The demo backend uses an A10G and is intended for interactive retrieval, not
  full evaluation.
- The first run for a model/split combination can be slow because it builds
  caches; repeated runs should reuse the Modal volume.
