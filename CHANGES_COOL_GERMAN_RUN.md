# Changes behind the "Cool German Run"

This document lists the code changes (from `main` → current branch) that were
actually on the execution path of the German (DE) evaluation with
`Qwen3-Reranker-8B` on Modal B200. Changes that exist in the diff but were
**not** part of this run (RF fuser, Harrier LoRA, BGE-reranker-v2-m3, BGE
fine-tuning pipeline, Gemma reranker, demo profile, tests) are deliberately
omitted below.

## What ran

- Modal app: `checkthat-evaluation-pipeline::evaluate_pipeline`
- GPU: `B200` (192 GB HBM, new-gen drivers)
- Dataset: CheckThat dev split, `--languages-csv de`, 386 queries
- Retrievers (enabled):
  1. **Harrier-27B** dense retriever (`microsoft/harrier-oss-v1-27b`)
  2. **BM25+ sparse retriever** with field-weighted translated queries
- Fusion: **RRF**, `fusion_top_k=100`, weights `(dense=0.8, sparse=0.2)`
- Reranker: **`Qwen3/Qwen3-Reranker-8B`** (cross-encoder, yes/no logit softmax)
- `final_top_k=5`
- Launch:
  ```bash
  uv run modal run -d -m clef_pipeline.main::main \
      --languages-csv de \
      --reranker-name qwen3-reranker-8b \
      --force-recompute-sparse-cache \
      --force-recompute-dense-documents \
      --force-recompute-dense-queries
  ```

## Changes grouped by stage

### 1. Dense retriever: Harrier-27B with OOM-safe defaults

`src/clef_pipeline/clef_pipeline/retrievers.py`

- `HarrierRetriever` added, wrapping `microsoft/harrier-oss-v1-27b`.
- `max_seq_length=1024` added to the constructor and to the cache key, so
  document/query embeddings are namespaced by context length
  (`microsoft--harrier-oss-v1-27b--msl1024/...`).
- `batch_size=1` default prevents the sliding-window causal mask from OOMing
  on A100/B200 even after the `expandable_segments` allocator fix.
- `torch_dtype="auto"` passthrough so bf16 weights load cleanly on B200.
- `_load_or_encode` caches embeddings per `(model_name, max_seq_length,
  doc_hash)` and re-uses them across runs.

### 2. Sparse retriever: multilingual BM25+ upgrade

`src/clef_pipeline/clef_pipeline/retrievers.py` + new
`src/clef_pipeline/clef_pipeline/translation_cache.py`

- `SparseRetriever` rewrite:
  - BM25+ with tunable `bm25_k1` / `bm25_b` (defaults 1.5 / 0.75).
  - **Field-weighted** document text: title ×3, abstract ×1, authors ×1,
    venue ×1 (replaces the legacy `title*8` stub).
  - **Per-language Snowball stemmer**: German → `SnowballStemmer("german")`,
    French → `SnowballStemmer("french")`, English → Lancaster (unchanged).
  - **Bigrams** on top of unigrams for tokenisation.
  - **Query translation** into English via `deep-translator`, cached via
    `TranslationCache` so repeat runs don't re-hit the network.
  - Cache tag includes every one of these knobs so upstream changes
    invalidate the correct caches automatically
    (`sparse/bm25plus-k1_1.50-b_0.75-stem_pl_v2-bigrams_1-fields_abstract1-authors1-title3-venue1-translate_v2/...`).
- `TranslationCache` (new 143-line module): JSON-backed, per-language,
  serialised with a file lock so multiple Modal workers don't corrupt it.

### 3. Fusion: RRF with configurable top-k and weights

`src/clef_pipeline/clef_pipeline/fusions.py` +
`src/clef_pipeline/clef_pipeline/pipeline_config.py`

- `PipelineConfig.fusion_top_k` raised from 30 → **100** so the reranker
  sees the top-100 candidates (empirical boost: DE fusion recall@100 jumps
  meaningfully, closing the "gold doc not in candidate pool" gap).
- `fusion_weights=(0.8, 0.2)` default: dense-heavy because dense MRR@5
  (~0.69) clearly beats sparse (~0.53) on this task.
- `ReciprocalRankFuser` parameterised on the weight vector so the same
  class handles equal-weight and weighted RRF.
- `build_pipeline_config("evaluation", fusion_method="rrf",
  fusion_weights=(0.8, 0.2), fusion_top_k=100, ...)` is what the DE run used
  via `main.py` defaults.

### 4. Reranker: Qwen3-Reranker-8B (this session)

`src/clef_pipeline/clef_pipeline/rerankers.py` +
`src/clef_pipeline/clef_pipeline/registry.py`

- New `Qwen3Reranker` class following the official Qwen3 inference recipe:
  - Loads `AutoModelForCausalLM` with `torch_dtype=torch.float16`,
    `device_map="auto"`, `padding_side="left"`.
  - Instruction-aware prompt template:
    ```
    <|im_start|>system
    Judge whether the Document meets the requirements based on the Query
    and the Instruct provided. Note that the answer can only be "yes" or "no".
    <|im_end|>
    <|im_start|>user
    <Instruct>: {instruction}
    <Query>: {query}
    <Document>: {doc}
    <|im_end|>
    <|im_start|>assistant
    <think>

    </think>

    ```
  - Default instruction tuned for this task:
    `"Given a scientific claim or social media post, retrieve the
    scientific paper that the claim refers to or is supported by."`
  - Score = `softmax([logit(no), logit(yes)])[:, 1]` at the last token, as
    prescribed by Qwen's model card.
  - `max_length=2048` (Qwen3 supports 32k but 2k is plenty for
    title+abstract and much faster).
  - `micro_batch_size=8` default, tuned for B200 with the 8B variant at
    fp16; safely drop to 2 on A100-40GB.
- Registry entries added: `qwen3-reranker-0.6b`, `qwen3-reranker-4b`,
  `qwen3-reranker-8b`. The DE run used the 8B variant.

### 5. Modal runtime: B200, bigger timeout, per-language checkpointing

`src/clef_pipeline/clef_pipeline/main.py`

- GPU set to **`B200`**. Modal's A100-80GB pool shipped with an older
  NVIDIA driver that the current torch wheel rejected ("NVIDIA driver on
  your system is too old (found version 12080)"), so we moved to B200 to
  eliminate that class of failure.
- Timeout raised from 3 h → **10 h** to fit Qwen3-Reranker-8B across all
  three languages without truncation (DE alone ran comfortably in <2 h).
- `transformers>=4.41,<4.55` pin in the Modal image (Qwen3 arch was added
  in 4.51; `<4.55` avoids the vmap sliding-window mask regression that OOMs
  Harrier-27B).
- Per-language **checkpointing**: after each language's query loop
  completes, the pipeline:
  1. Prints the per-language summary immediately (so metrics aren't hidden
     behind a late-run timeout).
  2. Writes `metrics_<lang>.json` to
     `/cache/embeddings/partial_results/<split>-<run_id>/`.
  3. Runs `embedding_cache.commit()` so partial results survive a crash.
- New CLI flag **`--languages-csv`** on both `evaluate_pipeline` and the
  `main` local entrypoint. Accepts a comma-separated subset of `{de, fr,
  en}` and filters the evaluation loop — this is what made the
  DE-only run possible without touching the code.
- New CLI flags **`--reranker-name`** and **`--reranker-model-name`** so
  swapping in `qwen3-reranker-8b` doesn't require a code change. Passing
  the empty string disables reranking entirely.

### 6. Pipeline glue

`src/clef_pipeline/clef_pipeline/pipeline.py` +
`src/clef_pipeline/clef_pipeline/registry.py`

- `RetrievalPipeline` now owns a **reranker micro-batch** helper so top-100
  candidate pools don't OOM when a reranker has a small `micro_batch_size`.
- `build_pipeline_from_config` wires up the `Qwen3Reranker` and picks up
  the CLI-provided `reranker_name`/`reranker_params` overrides.
- `RerankerConfig` accepts `name=None, enabled=False` for the
  retrieval-only case, which the language filter path relies on to avoid
  accidentally loading a reranker when it isn't needed.

## Files touched that were on the critical path

| File | Role in the DE run |
| ---- | ------------------ |
| `src/clef_pipeline/clef_pipeline/main.py` | Modal entrypoint, B200, timeout, `--languages-csv`, `--reranker-name`, per-language checkpointing |
| `src/clef_pipeline/clef_pipeline/pipeline_config.py` | Default `fusion_top_k=100`, `(0.8, 0.2)` RRF weights, reranker override plumbing |
| `src/clef_pipeline/clef_pipeline/pipeline.py` | Reranker micro-batch loop, top-k wiring |
| `src/clef_pipeline/clef_pipeline/registry.py` | `qwen3-reranker-8b` registered, `evaluation` profile wires `harrier-27b` + `sparse` |
| `src/clef_pipeline/clef_pipeline/retrievers.py` | `HarrierRetriever`, field-weighted BM25+, Snowball stemmer, bigrams, translated queries |
| `src/clef_pipeline/clef_pipeline/rerankers.py` | New `Qwen3Reranker` class (instruction-aware, yes/no softmax) |
| `src/clef_pipeline/clef_pipeline/translation_cache.py` | Sparse-retriever translation cache |
| `src/clef_pipeline/clef_pipeline/fusions.py` | Weighted RRF fuser |

## Files intentionally omitted from this PR

The exploratory branch (`T1Cursor`) also contained the items below. They
were **not** on the DE/FR/EN run's code path and have been dropped from
this clean PR branch; the originals remain on `T1Cursor` for future
ablation work:

- `src/clef_training/clef_training/mine_pipeline_negatives.py`
- `src/clef_training/clef_training/train_bge_reranker_modal.py`
- `src/clef_training/clef_training/train_harrier_modal.py`
- `src/clef_pipeline/clef_pipeline/bm25_grid_search.py`
- `src/clef_pipeline/clef_pipeline/diagnose_errors.py`
- `src/clef_pipeline/clef_pipeline/fusions.py::WeightedScoreFuser`
- `src/clef_pipeline/clef_pipeline/rerankers.py::BGEReranker`
- Registry entries `bge-reranker-v2-m3`, `qwen3-reranker-0.6b`,
  `qwen3-reranker-4b`, and the generic `sentence-transformer` /
  `e5-multilingual-large` / `jina-v3` retrievers

Reranker registry entries that pre-existed on `main` (`nemotron`,
`gemma2b`) and the `RandomForestFuser` are preserved untouched.
