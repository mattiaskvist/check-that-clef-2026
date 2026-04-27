# CLEF Pipeline (`src/clef_pipeline`)

This package contains the full evaluation pipeline. It combines dense retrieval with sparse BM25 retrieval, applies RRF fusion, and optionally reranks with a cross-encoder. The entrypoint is `clef_pipeline.main`.

## Package file map

Core module files under `src/clef_pipeline/clef_pipeline/`:

- `main.py` — Modal app definition, remote evaluation function, and local CLI entrypoint used by `uv run modal run -m clef_pipeline.main`.
- `pipeline.py` — `RetrievalPipeline` orchestration class: collection/query indexing, retriever execution, candidate fusion, reranking, and search output shaping.
- `pipeline_config.py` — frozen dataclass config objects (`RetrieverConfig`, `RerankerConfig`, `PipelineConfig`) plus preset profile builder.
- `registry.py` — string-to-component factory functions (`create_retriever`, `create_reranker`) and pipeline assembly helper.
- `retrievers.py` — dense retrievers (BGE-M3, Harrier) and sparse BM25+ retriever including embedding/query cache behavior.
- `rerankers.py` — cross-encoder rerankers that score fusion candidates and return sorted `(doc_index, score)` tuples.
- `metrics.py` — metric accumulator for multilingual evaluation, producing per-language and global summaries.
- `submission.py` — Codabench TSV writing utilities and Modal volume path/download command helpers.
- `interfaces.py` — abstract base contracts for retriever/reranker implementations.
- `logging_utils.py` — shared logger initialization and elapsed-time helper.
- `utils.py` — shared constants, ranking utilities (`MRR_at_5`, `recall_at_K`), and reciprocal-rank-fusion processor.
- `__init__.py` — package-level export list.

## How to run

1. Sign up for a Modal account at https://modal.com/, install the Modal CLI and log in.

```bash
uv run modal setup
```

2. Ensure you have your Hugging Face API token stored as a Modal Secret named "hf-token". You can create this secret using the Modal CLI:

```bash
uv run modal secret create hf-token HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

3. Ensure you have access to required Hugging Face model repos (for configured retrievers/rerankers).

```bash
# from root of the project, run:
uv run modal run -d -m clef_pipeline.main
```

## How the Modal workflow works

- `uv run modal run ...` starts your **local entrypoint** on your machine, and that entrypoint launches the heavy retrieval/reranking work as a **remote Modal function**.
- The expensive compute (GPU, model inference, indexing) runs in Modal's cloud containers, not on your laptop.
- With `-d`, the app is not stopped if your local process dies or disconnects.

### Do you need to keep your computer running?

- For pure evaluation runs (`uv run modal run -d -m clef_pipeline.main`), you can treat it as fire-and-forget once it's started.
- For submission export, this project writes `predictions_{lang}.tsv` to the Modal volume and prints a `modal volume get ...` command you can run locally to download them.

Sparse query rankings and scores are cached between runs (under the existing Modal volume mount), so reranking and fusion experiments can iterate without recomputing BM25 for each query.

If you need to rebuild sparse cache artifacts after code changes, run:

```bash
uv run modal run -d -m clef_pipeline.main --force-recompute-sparse-cache
```

Dense embeddings are also cached between runs. You can force dense recomputation independently:

```bash
# Recompute dense document embeddings
uv run modal run -d -m clef_pipeline.main --force-recompute-dense-documents

# Recompute dense query embeddings
uv run modal run -d -m clef_pipeline.main --force-recompute-dense-queries
```

## Export submission TSV files

To generate Codabench-ready `predictions_{lang}.tsv` files (columns: `index`, `preds`), run:

```bash
uv run modal run -m clef_pipeline.main \
  --split dev \
  --export-submission-tsv \
  --submission-volume-subdir submissions \
  --submission-download-dir submissions
```

For competition export, switch to the unlabeled split:

```bash
uv run modal run -m clef_pipeline.main \
  --split test \
  --export-submission-tsv \
  --submission-volume-subdir submissions \
  --submission-download-dir submissions
```

The official `test` split does not include `pubkey`, so this mode is submission-only:
the pipeline will generate predictions, but it will not compute local metrics or
write a metrics JSON file.

You can also run on `train` for debugging or analysis:

```bash
uv run modal run -m clef_pipeline.main \
  --split train \
  --export-submission-tsv \
  --submission-volume-subdir submissions \
  --submission-download-dir submissions
```

This writes one TSV per language (for example `predictions_en.tsv`, `predictions_de.tsv`, `predictions_fr.tsv`) into the Modal volume under:

`/submissions/<split>-<timestamp>`

At the end of the run, the command prints an exact download command you can run locally, for example:

```bash
uv run modal volume get checkthat-embedding-cache /submissions/dev-20260416-191700 submissions
```

## Prepare final `predictions.zip` for Codabench

After exporting with `--split test` and downloading the folder, package the TSVs like this:

```bash
cd submissions/test-<timestamp>
ls predictions_*.tsv
zip -r predictions.zip predictions_*.tsv
unzip -l predictions.zip
```

Expected contents:
- `predictions_en.tsv`
- `predictions_de.tsv`
- `predictions_fr.tsv`

Upload `predictions.zip` to the Codabench competition page.

## Current Stats on Dev

Reported metrics (MRR@5, R@5, R@10, R@30) for the full pipeline on the dev set are as follows:

| Language / Group | n Tweets | Dense (MRR@5 / R@5 / R@10 / R@30) | Sparse (MRR@5 / R@5 / R@10 / R@30) | RRF (MRR@5 / R@5 / R@10 / R@30) | Rerank (MRR@5 / R@5 / R@10 / R@30) |
|------------------|----------|-----------------------------------|------------------------------------|---------------------------------|------------------------------------|
| DE               | 386      | 0.5918 / 0.6891 / 0.7487 / 0.8394 | 0.4048 / 0.5000 / 0.5699 / 0.6477  | 0.5315 / - / - / 0.8109         | 0.6247 / 0.6995 / - / -            |
| FR               | 702      | 0.7053 / 0.7977 / 0.8490 / 0.8832 | 0.5446 / 0.6154 / 0.6538 / 0.7336  | 0.6487 / - / - / 0.8604         | 0.7240 / 0.8077 / - / -            |
| EN               | 3905     | 0.6946 / 0.7972 / 0.8371 / 0.8960 | 0.5388 / 0.6151 / 0.6553 / 0.7168  | 0.6271 / - / - / 0.8574         | 0.7332 / 0.8038 / - / -            |
| GLOBAL AVERAGE   | 4993     | 0.6882 / 0.7889 / 0.8320 / 0.8898 | 0.5293 / 0.6062 / 0.6485 / 0.7138  | 0.6228 / - / - / 0.8542         | 0.7235 / 0.7963 / - / -            |

Note: These numbers will need to be updated as we continue to refine the pipeline. The current results are based on the dev set, which we are treating as a validation set for iterative improvements. Dashing indicates metrics not logged for that specific pipeline stage.

## Streamlit demo package

The Streamlit app now lives in `src/clef_demo/` and imports the pipeline package.

### Local Streamlit demo

```bash
uv sync
uv run streamlit run src/clef_demo/clef_demo/streamlit_app.py
```

The demo:
- indexes the full collection;
- optionally merges custom JSON documents by `pubkey` (custom overrides base);
- allows selecting one or more retrievers;
- allows enabling/disabling fusion and reranking;
- returns top-5 matches for a user tweet.

### Host Streamlit demo on Modal

```bash
uv run modal serve -m clef_demo.modal_app
```

To deploy persistently:

```bash
uv run modal deploy -m clef_demo.modal_app
```
