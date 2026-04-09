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

Sparse query rankings and scores are cached between runs (under the existing Modal volume mount), so reranking and fusion experiments can iterate without recomputing BM25 for each query.

If you need to rebuild sparse cache artifacts after code changes, run:

```bash
uv run modal run -d -m full_pipeline.main --force-recompute-sparse-cache
```

## Current Stats on Dev

Reported MRR@5 for the full pipeline on the dev set (combining dense retrieval, sparse retrieval, RRF fusion, and final re-ranking) is as follows:

| Language / Group | n Tweets | Dense  | Sparse | RRF    | Rerank |
|------------------|----------|--------|--------|--------|--------|
| DE               | 386      | 0.4945 | 0.4047 | 0.5216 | 0.5459 |
| FR               | 702      | 0.5704 | 0.5446 | 0.6194 | 0.6557 |
| EN               | 3905     | 0.5600 | 0.5350 | 0.5937 | 0.6277 |
| GLOBAL AVERAGE   | 4993     | 0.5564 | 0.5263 | 0.5918 | 0.6253 |

Note: These numbers will need to be updated as we continue to refine the pipeline. The current results are based on the dev set, which we are treating as a validation set for iterative improvements.

## Next Steps

- Continue to iterate on the dense retrieval fine-tuning to try to close the gap between the dense-only and final reranked results. This may involve experimenting with different training hyperparameters, more training epochs, or even trying out different base models. Most likely, the hard negative mining strategy can be improved to provide more challenging negatives during training, which should help the model learn better representations.

- Explore more advanced fusion techniques beyond RRF, such as learning-to-rank models that can take the dense and sparse scores as input features and learn an optimal way to combine them.

- For the final re-ranking stage, we can experiment with different cross-encoder architectures or even try out generative re-ranking approaches using large language models to see if we can further boost the final retrieval performance. I belive [jina-reranker-v3](https://huggingface.co/jina-ai/jina-reranker-v3) could be a strong candidate for this task, as it is specifically designed for re-ranking and has shown strong performance on various benchmarks. The current cross-encoder we are using ([BAAI/bge-reraker-v2-gemma](https://huggingface.co/BAAI/bge-reraker-v2-gemma)) is a good starting point, but it may not be fully optimized for our specific retrieval task, with different languages etc. It would be worth experimenting with other models to see if we can achieve better final results. Maybe the newly released [Gemma 4](https://deepmind.google/models/gemma/gemma-4/) can be a strong candidate for this, given its state-of-the-art performance on various NLP tasks.

- Improve the sparse retrieval component by experimenting with different BM25 parameters or even trying out more advanced sparse retrieval models like SPLADE or DeepCT, which can provide better sparse representations and potentially improve the overall fusion results.

- Work on how metadata from the documents can be better utilized in the retrieval and re-ranking process. Currently, the pipeline only uses the title and abstract for retrieval, but `authors` and `venue` information could also be valuable signals for both retrieval and re-ranking. We can experiment with ways to incorporate this metadata, such as concatenating it with the title and abstract for the dense retriever, or using it as additional features in the re-ranking stage. Maybe we can use the information to look for matches and boost scores for documents that have the same authors or are published in the same venue as the query paper, as these could be strong indicators of relevance.
