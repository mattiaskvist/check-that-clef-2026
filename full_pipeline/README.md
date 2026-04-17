# Full Evaluation Pipeline

This directory contains the code for the full evaluation pipeline, which integrates the dense retrieval model (BGE-M3 with LoRA fine-tuning) with the sparse retrieval results (vanilla BM25), applies RRF fusion, and performs final re-ranking using a cross-encoder. The main script `main.py` orchestrates the entire process.

## How to run

1. Sign up for a Modal account at https://modal.com/, install the Modal CLI and log in.

```bash
uv run modal setup
```

2. Ensure you have your Hugging Face API token stored as a Modal Secret named "hf-token". You can create this secret using the Modal CLI:

```bash
uv run modal secret create hf-token HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

3. Ensure you have access to the huggingface repo "boyes-boys-clef-2026/bge-m3-checkthat-finetuned" which contains the LoRA adapter weights for the fine-tuned BGE-M3 model.

```bash
# from root of the project, run:
uv run modal run -d -m full_pipeline.main
```

## How the Modal workflow works

- `uv run modal run ...` starts your **local entrypoint** on your machine, and that entrypoint launches the heavy retrieval/reranking work as a **remote Modal function**.
- The expensive compute (GPU, model inference, indexing) runs in Modal's cloud containers, not on your laptop.
- With `-d`, the app is not stopped if your local process dies or disconnects.

### Do you need to keep your computer running?

- For pure evaluation runs (`uv run modal run -d -m full_pipeline.main`), you can treat it as fire-and-forget once it's started.
- For submission export, this project writes `predictions_{lang}.tsv` to the Modal volume and prints a `modal volume get ...` command you can run locally to download them.

Sparse query rankings and scores are cached between runs (under the existing Modal volume mount), so reranking and fusion experiments can iterate without recomputing BM25 for each query.

If you need to rebuild sparse cache artifacts after code changes, run:

```bash
uv run modal run -d -m full_pipeline.main --force-recompute-sparse-cache
```

Dense embeddings are also cached between runs. You can force dense recomputation independently:

```bash
# Recompute dense document embeddings
uv run modal run -d -m full_pipeline.main --force-recompute-dense-documents

# Recompute dense query embeddings
uv run modal run -d -m full_pipeline.main --force-recompute-dense-queries
```

## Export submission TSV files

To generate Codabench-ready `predictions_{lang}.tsv` files (columns: `index`, `preds`), run:

```bash
uv run modal run -m full_pipeline.main \
  --split dev \
  --export-submission-tsv \
  --submission-volume-subdir submissions \
  --submission-download-dir submissions
```

For competition export, switch to the unlabeled split:

```bash
uv run modal run -m full_pipeline.main \
  --split test \
  --export-submission-tsv \
  --submission-volume-subdir submissions \
  --submission-download-dir submissions
```

You can also run on `train` for debugging or analysis:

```bash
uv run modal run -m full_pipeline.main \
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

## Current Stats on Dev

Reported metrics (MRR@5, R@5, R@10, R@30) for the full pipeline on the dev set are as follows:

| Language / Group | n Tweets | Dense (MRR@5 / R@5 / R@10 / R@30) | Sparse (MRR@5 / R@5 / R@10 / R@30) | RRF (MRR@5 / R@5 / R@10 / R@30) | Rerank (MRR@5 / R@5 / R@10 / R@30) |
|------------------|----------|-----------------------------------|------------------------------------|---------------------------------|------------------------------------|
| DE               | 386      | 0.5918 / 0.6891 / 0.7487 / 0.8394 | 0.4048 / 0.5000 / 0.5699 / 0.6477  | 0.5315 / - / - / 0.8109         | 0.6247 / 0.6995 / - / -            |
| FR               | 702      | 0.7053 / 0.7977 / 0.8490 / 0.8832 | 0.5446 / 0.6154 / 0.6538 / 0.7336  | 0.6487 / - / - / 0.8604         | 0.7240 / 0.8077 / - / -            |
| EN               | 3905     | 0.6946 / 0.7972 / 0.8371 / 0.8960 | 0.5388 / 0.6151 / 0.6553 / 0.7168  | 0.6271 / - / - / 0.8574         | 0.7332 / 0.8038 / - / -            |
| GLOBAL AVERAGE   | 4993     | 0.6882 / 0.7889 / 0.8320 / 0.8898 | 0.5293 / 0.6062 / 0.6485 / 0.7138  | 0.6228 / - / - / 0.8542         | 0.7235 / 0.7963 / - / -            |

Note: These numbers will need to be updated as we continue to refine the pipeline. The current results are based on the dev set, which we are treating as a validation set for iterative improvements. Dashing indicates metrics not logged for that specific pipeline stage.

## Next Steps

- Continue to iterate on the dense retrieval fine-tuning to try to close the gap between the dense-only and final reranked results. This may involve experimenting with different training hyperparameters, more training epochs, or even trying out different base models. Most likely, the hard negative mining strategy can be improved to provide more challenging negatives during training, which should help the model learn better representations.

- Explore more advanced fusion techniques beyond RRF, such as learning-to-rank models that can take the dense and sparse scores as input features and learn an optimal way to combine them.

- For the final re-ranking stage, we can experiment with different cross-encoder architectures or even try out generative re-ranking approaches using large language models to see if we can further boost the final retrieval performance. I belive [jina-reranker-v3](https://huggingface.co/jina-ai/jina-reranker-v3) could be a strong candidate for this task, as it is specifically designed for re-ranking and has shown strong performance on various benchmarks. The current cross-encoder we are using ([BAAI/bge-reraker-v2-gemma](https://huggingface.co/BAAI/bge-reraker-v2-gemma)) is a good starting point, but it may not be fully optimized for our specific retrieval task, with different languages etc. It would be worth experimenting with other models to see if we can achieve better final results. Maybe the newly released [Gemma 4](https://deepmind.google/models/gemma/gemma-4/) can be a strong candidate for this, given its state-of-the-art performance on various NLP tasks.

- Improve the sparse retrieval component by experimenting with different BM25 parameters or even trying out more advanced sparse retrieval models like SPLADE or DeepCT, which can provide better sparse representations and potentially improve the overall fusion results.

- Work on how metadata from the documents can be better utilized in the retrieval and re-ranking process. Currently, the pipeline only uses the title and abstract for retrieval, but `authors` and `venue` information could also be valuable signals for both retrieval and re-ranking. We can experiment with ways to incorporate this metadata, such as concatenating it with the title and abstract for the dense retriever, or using it as additional features in the re-ranking stage. Maybe we can use the information to look for matches and boost scores for documents that have the same authors or are published in the same venue as the query paper, as these could be strong indicators of relevance.
