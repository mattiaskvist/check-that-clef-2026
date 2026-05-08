"""Modal entrypoint for multilingual evaluation and submission export."""

import modal

from .logging_utils import StageTimer, get_logger
from .metrics import DENSE_SPARSE_RECALL_CUTOFFS, EvaluationMetrics
from .pipeline_config import build_pipeline_config
from .registry import build_pipeline_from_config
from .submission import (
    SUBMISSION_TOP_K,
    modal_volume_download_command,
    normalize_split,
    submission_volume_remote_dir,
    write_submission_tsv_files,
)
from .utils import CHECKTHAT_DATASET, load_query_split, read_custom_papers

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "sentence-transformers <= 5.3.0",
        "peft",
        "rank_bm25",
        "datasets",
        "tqdm",
        "transformers <= 5.5.1",
        "accelerate",
        "nltk",
        "Pillow",
        "torchvision",
        "deep-translator",
        "huggingface-hub <=	1.9.2",
    )
)

app = modal.App("checkthat-evaluation-pipeline")
EMBEDDING_CACHE_VOLUME_NAME = "checkthat-embedding-cache"
embedding_cache = modal.Volume.from_name(
    EMBEDDING_CACHE_VOLUME_NAME, create_if_missing=True
)
CACHE_MOUNT = "/cache/embeddings"
HF_CACHE_DIR = f"{CACHE_MOUNT}/huggingface"
logger = get_logger("clef_pipeline.modal")


def _configure_hf_cache() -> None:
    """Point Hugging Face downloads at the shared Modal volume."""
    import os

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HF_HUB_CACHE", f"{HF_CACHE_DIR}/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{HF_CACHE_DIR}/transformers")


def _reranker_hf_model_id(reranker_model: str | None) -> str | None:
    """Resolve a registry reranker name to its Hugging Face model id."""
    if not reranker_model:
        return None
    model_ids = {
        "nemotron": "nvidia/llama-nemotron-rerank-1b-v2",
        "gemma2b": "BAAI/bge-reranker-v2-gemma",
        "jina-v3": "jinaai/jina-reranker-v3",
        "qwen3-reranker-8b": "Qwen/Qwen3-Reranker-8B",
        "qwen3-reranker-4b": "Qwen/Qwen3-Reranker-4B",
        "qwen3-reranker-0.6b": "Qwen/Qwen3-Reranker-0.6B",
    }
    return model_ids.get(reranker_model)


def _format_dense_sparse_metrics(stage_metrics: dict[str, float]) -> str:
    """Format MRR and recall metrics for dense/sparse stages."""
    recall_metrics = " | ".join(
        f"R@{cutoff}: {stage_metrics[f'r{cutoff}']:.4f}"
        for cutoff in DENSE_SPARSE_RECALL_CUTOFFS
    )
    return f"MRR@5: {stage_metrics['mrr5']:.4f} | {recall_metrics}"


def _print_language_summary(
    lang: str, data: dict[str, object], fusion_top_k: int, fusion_label: str
):
    """Print a formatted per-language metric summary to stdout.

    Args:
        lang: Language code.
        data: Summary payload produced by ``EvaluationMetrics.summary``.
        fusion_top_k: Fusion cutoff used for fusion recall display.
        fusion_label: Label used when printing fusion-stage metrics.
    """
    metrics = data["metrics"]
    if metrics is None:
        print(f"\n--- Summary for {lang.upper()} ({data['Total Queries']} Queries) ---")
        print(
            f"No labels available; generated predictions only. "
            f"Labeled: {data['Labeled Queries']}, Unlabeled: {data['Unlabeled Queries']}."
        )
        return

    print(f"\n--- Summary for {lang.upper()} ({data['Total Queries']} Queries) ---")
    print(f"Dense Only:   {_format_dense_sparse_metrics(metrics['dense'])}")
    print(f"Sparse Only:  {_format_dense_sparse_metrics(metrics['sparse'])}")
    print(
        f"{fusion_label} Output:   MRR@5: {metrics['rrf']['mrr5']:.4f} | R@{fusion_top_k}: {metrics['rrf'][f'r{fusion_top_k}']:.4f}"
    )
    print(
        f"Final Rerank: MRR@5: {metrics['final']['mrr5']:.4f} | R@5: {metrics['final']['r5']:.4f}"
    )


def _validate_metrics_output_request(
    split: str, metrics_output_file: str | None
) -> None:
    """Reject metrics export for splits without labels."""
    if split == "test" and metrics_output_file:
        raise ValueError(
            "metrics_output_file is not supported for split='test' because the "
            "official test set has no pubkey labels."
        )


def _record_query_result(
    *,
    lang: str,
    row: dict,
    query_idx: int,
    result: dict[str, object],
    metrics: EvaluationMetrics,
    submission_predictions: dict[str, list[dict[str, object]]] | None = None,
) -> None:
    """Update submission payloads and metrics for one evaluated query."""
    if submission_predictions is not None:
        submission_predictions[lang].append(
            {
                "index": row.get("index", query_idx),
                "preds": result["preds"][:SUBMISSION_TOP_K],
            }
        )

    metrics.add_query(
        lang=lang,
        true_pubkey=row.get("pubkey"),
        stages=result["stages"],
    )


def _chunk_indices(total: int, chunk_size: int) -> list[tuple[int, int]]:
    """Split a query range into half-open chunks."""
    effective_chunk_size = max(1, int(chunk_size))
    return [
        (start, min(start + effective_chunk_size, total))
        for start in range(0, total, effective_chunk_size)
    ]


def _split_indices(total: int, num_splits: int) -> list[list[int]]:
    """Split query indices into at most ``num_splits`` balanced groups."""
    if total <= 0:
        return []
    split_count = min(total, max(1, int(num_splits)))
    base_size, remainder = divmod(total, split_count)
    groups = []
    start = 0
    for split_idx in range(split_count):
        size = base_size + (1 if split_idx < remainder else 0)
        end = start + size
        groups.append(list(range(start, end)))
        start = end
    return groups


def _parse_query_shard(query_shard: str | None) -> tuple[int, int] | None:
    """Parse a one-based query shard spec like ``1/3``."""
    if not query_shard:
        return None
    try:
        shard_index_text, shard_count_text = query_shard.split("/", 1)
        shard_index = int(shard_index_text)
        shard_count = int(shard_count_text)
    except ValueError as exc:
        raise ValueError("query_shard must use the form '<index>/<count>'.") from exc
    if shard_count < 1:
        raise ValueError("query_shard count must be at least 1.")
    if shard_index < 1 or shard_index > shard_count:
        raise ValueError("query_shard index must be between 1 and count.")
    return shard_index, shard_count


def _query_shard_indices(total: int, query_shard: str | None) -> list[int]:
    """Return deterministic contiguous query indices for one shard."""
    parsed_shard = _parse_query_shard(query_shard)
    if parsed_shard is None:
        return list(range(total))
    shard_index, shard_count = parsed_shard
    groups = _split_indices(total, shard_count)
    if shard_index > len(groups):
        return []
    return groups[shard_index - 1]


def _submission_split_label(split: str, query_shard: str | None) -> str:
    """Return the submission directory split label, including shard if present."""
    if not query_shard:
        return split
    return f"{split}-shard-{query_shard.replace('/', 'of')}"


def _evaluate_query_chunk(
    *,
    pipeline,
    lang: str,
    tweets: list[dict],
    query_indices: list[int],
    cache_lang: str,
) -> list[dict[str, object]]:
    """Evaluate a query slice and return replayable per-query results."""
    import time

    from tqdm import tqdm

    started_at = time.monotonic()
    outputs = []
    if query_indices:
        query_range = f"{min(query_indices)}-{max(query_indices)}"
    else:
        query_range = "empty"
    progress_desc = f"Worker {lang.upper()} queries {query_range}"
    for query_idx in tqdm(
        query_indices,
        desc=progress_desc,
        unit="query",
        position=0,
        leave=True,
    ):
        row = tweets[query_idx]
        result = pipeline.search_cached_query(
            query_idx=query_idx,
            query_text=row["text"],
            lang=lang,
            cache_lang=cache_lang,
        )
        outputs.append(
            {
                "lang": lang,
                "query_idx": query_idx,
                "row": row,
                "result": result,
            }
        )
    elapsed_seconds = time.monotonic() - started_at
    for output in outputs:
        output["chunk_elapsed_seconds"] = elapsed_seconds
        output["chunk_query_count"] = len(query_indices)
    return outputs


@app.function(
    image=image,
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={CACHE_MOUNT: embedding_cache},
)
def preload_reranker_model(reranker_model: str | None, disable_reranker: bool) -> None:
    """Download reranker weights once into the shared Hugging Face cache."""
    if disable_reranker:
        return

    _configure_hf_cache()
    model_id = _reranker_hf_model_id(reranker_model)
    if model_id is None:
        return

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_id)
    embedding_cache.commit()


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 10,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={CACHE_MOUNT: embedding_cache},
)
def evaluate_query_chunk(payload: dict[str, object]) -> list[dict[str, object]]:
    """Evaluate one cached query chunk in its own GPU container."""
    import os

    from datasets import load_dataset

    _configure_hf_cache()
    split = normalize_split(payload["split"])
    lang = payload["lang"]
    query_indices = payload["query_indices"]
    cache_lang = payload["cache_lang"]
    custom_papers = payload.get("custom_papers")

    sparse_vanilla = payload["sparse_vanilla"]
    sparse_k1 = 1.5 if sparse_vanilla else 2.5
    sparse_b = 0.75 if sparse_vanilla else 0.85
    sparse_use_bigrams = not sparse_vanilla
    sparse_use_translation = not sparse_vanilla

    config = build_pipeline_config(
        payload["profile"],
        fusion_method=payload["fusion_method"],
        fusion_top_k=payload["fusion_top_k"],
        hf_fusion_repo_id=payload["hf_fusion_repo_id"],
        hf_token=os.environ.get("HF_TOKEN"),
        dense_model=payload["dense_model"],
        disable_sparse=payload["disable_sparse"],
        reranker_model=payload["reranker_model"],
        disable_reranker=payload["disable_reranker"],
        sparse_k1=sparse_k1,
        sparse_b=sparse_b,
        sparse_use_bigrams=sparse_use_bigrams,
        sparse_use_translation=sparse_use_translation,
    )
    pipeline = build_pipeline_from_config(config)

    collection_dataset = load_dataset(
        CHECKTHAT_DATASET, "collection", split="collection"
    )
    collection_documents = collection_dataset.to_list()
    if custom_papers:
        custom_path = os.path.join(CACHE_MOUNT, custom_papers)
        collection_documents.extend(read_custom_papers(custom_path, 11000))

    pipeline.index_collection(
        collection_documents=collection_documents,
        cache_dir=CACHE_MOUNT,
        force_recompute_dense_documents=False,
    )

    tweets = load_query_split(lang, split)
    pipeline.index_queries_for_language(
        lang=lang,
        query_texts=[row["text"] for row in tweets],
        cache_lang=cache_lang,
        cache_dir=CACHE_MOUNT,
        force_recompute_sparse_cache=False,
        force_recompute_dense_queries=False,
    )
    pipeline.prepare_fusion_model(
        cache_dir=CACHE_MOUNT,
        languages=[lang],
        force_retrain_fusion=False,
        force_recompute_dense_queries=False,
        force_recompute_sparse_cache=False,
    )
    pipeline.unload_dense_models()

    return _evaluate_query_chunk(
        pipeline=pipeline,
        lang=lang,
        tweets=tweets,
        query_indices=query_indices,
        cache_lang=cache_lang,
    )


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 10,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={CACHE_MOUNT: embedding_cache},
)
def prepare_pipeline_caches(payload: dict[str, object]) -> dict[str, object]:
    """Run GPU-backed retrieval cache preparation before CPU orchestration."""
    import os

    from datasets import load_dataset

    _configure_hf_cache()
    split = normalize_split(payload["split"])
    languages = payload["languages"] or ["de", "fr", "en"]
    sparse_vanilla = payload["sparse_vanilla"]
    sparse_k1 = 1.5 if sparse_vanilla else 2.5
    sparse_b = 0.75 if sparse_vanilla else 0.85
    sparse_use_bigrams = not sparse_vanilla
    sparse_use_translation = not sparse_vanilla

    config = build_pipeline_config(
        payload["profile"],
        fusion_method=payload["fusion_method"],
        fusion_top_k=payload["fusion_top_k"],
        hf_fusion_repo_id=payload["hf_fusion_repo_id"],
        hf_token=os.environ.get("HF_TOKEN"),
        dense_model=payload["dense_model"],
        disable_sparse=payload["disable_sparse"],
        reranker_model=payload["reranker_model"],
        disable_reranker=True,
        sparse_k1=sparse_k1,
        sparse_b=sparse_b,
        sparse_use_bigrams=sparse_use_bigrams,
        sparse_use_translation=sparse_use_translation,
    )
    pipeline = build_pipeline_from_config(config)

    logger.info("Loading collection and building index...")
    collection_dataset = load_dataset(
        CHECKTHAT_DATASET, "collection", split="collection"
    )
    collection_documents = collection_dataset.to_list()
    custom_papers = payload.get("custom_papers")
    if custom_papers:
        custom_path = os.path.join(CACHE_MOUNT, custom_papers)
        logger.info(f"Appending custom papers from: {custom_path}")
        custom_docs = read_custom_papers(custom_path, 11000)
        collection_documents.extend(custom_docs)
        logger.info(f"Total documents after append: {len(collection_documents)}")
    pipeline.index_collection(
        collection_documents=collection_documents,
        cache_dir=CACHE_MOUNT,
        force_recompute_dense_documents=payload["force_recompute_dense_documents"],
    )
    embedding_cache.commit()

    cache_langs = {lang: f"{split}_{lang}" for lang in languages}
    for lang in languages:
        tweets = load_query_split(lang, split)
        pipeline.index_queries_for_language(
            lang=lang,
            query_texts=[row["text"] for row in tweets],
            cache_lang=cache_langs[lang],
            cache_dir=CACHE_MOUNT,
            force_recompute_sparse_cache=payload["force_recompute_sparse_cache"],
            force_recompute_dense_queries=payload["force_recompute_dense_queries"],
        )
        embedding_cache.commit()

    pipeline.prepare_fusion_model(
        cache_dir=CACHE_MOUNT,
        languages=languages,
        force_retrain_fusion=payload["force_retrain_fusion"],
        force_recompute_dense_queries=payload["force_recompute_dense_queries"],
        force_recompute_sparse_cache=payload["force_recompute_sparse_cache"],
        on_cache_update=embedding_cache.commit,
    )
    pipeline.unload_dense_models()
    embedding_cache.commit()
    return {"languages": languages, "cache_langs": cache_langs}


@app.function(
    image=image,
    timeout=60 * 60 * 10,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={CACHE_MOUNT: embedding_cache},
)
def evaluate_pipeline(
    force_recompute_sparse_cache: bool = False,
    force_recompute_dense_documents: bool = False,
    force_recompute_dense_queries: bool = False,
    split: str = "dev",
    collect_submission: bool = False,
    submission_volume_subdir: str = "submissions",
    fusion_method: str = "rrf",
    fusion_top_k: int = 30,
    force_retrain_fusion: bool = False,
    hf_fusion_repo_id: str | None = "boyes-boys-clef-2026/random-forest-fuser",
    profile: str = "custom",
    dense_model: str = "harrier-27b",
    disable_sparse: bool = False,
    reranker_model: str = "nemotron",
    disable_reranker: bool = False,
    sparse_vanilla: bool = False,
    custom_papers: str | None = None,
    languages: list[str] | None = None,
    rerank_parallelism: int = 1,
    rerank_chunk_size: int | None = None,
    query_shard: str | None = None,
):
    """Run the full retrieval evaluation workflow on Modal.

    Args:
        force_recompute_sparse_cache: Recompute sparse query cache even if present.
        force_recompute_dense_documents: Recompute dense document embeddings.
        force_recompute_dense_queries: Recompute dense query embeddings.
        split: Dataset split to evaluate (``train``, ``dev``, or ``test``).
        collect_submission: Whether to write submission TSV files.
        submission_volume_subdir: Subdirectory under cache volume for submissions.
        fusion_method: Fusion strategy (``rrf`` or ``random_forest``).
        force_retrain_fusion: Force retraining learned fusion model instead of loading cache.
        hf_fusion_repo_id: Hugging Face repository ID to push/pull fusion models.

    Returns:
        Global language results and optional submission artifact metadata.
    """
    from datetime import datetime, timezone

    from tqdm import tqdm

    timer = StageTimer()
    split = normalize_split(split)

    config = build_pipeline_config(
        profile,
        fusion_method=fusion_method,
        fusion_top_k=fusion_top_k,
        hf_fusion_repo_id=hf_fusion_repo_id,
        hf_token=None,
        dense_model=dense_model,
        disable_sparse=disable_sparse,
        reranker_model=reranker_model,
        disable_reranker=disable_reranker,
    )

    languages = languages or ["de", "fr", "en"]
    prep_payload = {
        "split": split,
        "languages": languages,
        "fusion_method": fusion_method,
        "fusion_top_k": fusion_top_k,
        "hf_fusion_repo_id": hf_fusion_repo_id,
        "profile": profile,
        "dense_model": dense_model,
        "disable_sparse": disable_sparse,
        "reranker_model": reranker_model,
        "sparse_vanilla": sparse_vanilla,
        "custom_papers": custom_papers,
        "force_recompute_sparse_cache": force_recompute_sparse_cache,
        "force_recompute_dense_documents": force_recompute_dense_documents,
        "force_recompute_dense_queries": force_recompute_dense_queries,
        "force_retrain_fusion": force_retrain_fusion,
    }
    prepare_result = prepare_pipeline_caches.remote(prep_payload)
    preload_reranker_model.remote(reranker_model, disable_reranker)

    languages = prepare_result["languages"]
    lang_tweets: dict[str, list[dict]] = {}
    cache_langs = prepare_result["cache_langs"]
    for lang in languages:
        tweets = load_query_split(lang, split)
        lang_tweets[lang] = tweets

    logger.info("Running multilingual evaluation...")
    fusion_label = "RF Fusion" if config.fusion_method == "random_forest" else "RRF"
    metrics = EvaluationMetrics(fusion_top_k=config.fusion_top_k)
    submission_predictions = (
        {lang: [] for lang in languages} if collect_submission else {}
    )

    chunk_payloads = []
    worker_count = max(1, int(rerank_parallelism))
    for lang in languages:
        print("\n==========================================")
        print(f"  QUEUING EVALUATION FOR LANGUAGE: {lang.upper()}")
        print("==========================================")
        lang_query_indices = _query_shard_indices(len(lang_tweets[lang]), query_shard)
        if query_shard:
            print(
                f"  Query shard {query_shard}: {len(lang_query_indices)} of "
                f"{len(lang_tweets[lang])} queries"
            )
        if rerank_chunk_size:
            query_chunks = [
                lang_query_indices[start:end]
                for start, end in _chunk_indices(
                    len(lang_query_indices), rerank_chunk_size
                )
            ]
        else:
            query_chunks = _split_indices(len(lang_query_indices), worker_count)
            query_chunks = [
                [lang_query_indices[idx] for idx in chunk] for chunk in query_chunks
            ]
        for query_indices in query_chunks:
            chunk_payloads.append(
                {
                    "split": split,
                    "lang": lang,
                    "query_indices": query_indices,
                    "cache_lang": cache_langs[lang],
                    "fusion_method": fusion_method,
                    "fusion_top_k": fusion_top_k,
                    "hf_fusion_repo_id": hf_fusion_repo_id,
                    "profile": profile,
                    "dense_model": dense_model,
                    "disable_sparse": disable_sparse,
                    "reranker_model": reranker_model,
                    "disable_reranker": disable_reranker,
                    "sparse_vanilla": sparse_vanilla,
                    "custom_papers": custom_papers,
                }
            )

    total_query_count = sum(len(payload["query_indices"]) for payload in chunk_payloads)
    with tqdm(
        total=total_query_count,
        desc="Evaluating queries",
        unit="query",
    ) as pbar:
        for wave_start in range(0, len(chunk_payloads), worker_count):
            wave = chunk_payloads[wave_start : wave_start + worker_count]
            for chunk_results in evaluate_query_chunk.map(wave, order_outputs=False):
                chunk_query_count = len(chunk_results)
                chunk_elapsed = 0.0
                if chunk_results:
                    chunk_query_count = chunk_results[0].get(
                        "chunk_query_count", chunk_query_count
                    )
                    chunk_elapsed = chunk_results[0].get("chunk_elapsed_seconds", 0.0)
                pbar.update(chunk_query_count)
                if chunk_elapsed > 0:
                    queries_per_second = chunk_query_count / chunk_elapsed
                    pbar.set_postfix(
                        {
                            "chunk": chunk_query_count,
                            "q/s": f"{queries_per_second:.2f}",
                        }
                    )
                for item in chunk_results:
                    _record_query_result(
                        lang=item["lang"],
                        row=item["row"],
                        query_idx=item["query_idx"],
                        result=item["result"],
                        metrics=metrics,
                        submission_predictions=(
                            submission_predictions if collect_submission else None
                        ),
                    )

    if collect_submission:
        for lang_predictions in submission_predictions.values():
            lang_predictions.sort(key=lambda row: row["index"])

    summary = metrics.summary()
    global_results = summary["languages"]
    for lang in languages:
        _print_language_summary(
            lang, global_results[lang], config.fusion_top_k, fusion_label
        )

    print("\n\n" + "*" * 80)
    print("*" + " FINAL MULTILINGUAL EVALUATION SUMMARY ".center(78) + "*")
    print("*" * 80)
    for lang in languages:
        data = global_results[lang]
        lang_metrics = data["metrics"]
        if lang_metrics is None:
            print(
                f"\n[{lang.upper()}] - {data['Total Queries']} Queries "
                f"({data['Unlabeled Queries']} unlabeled, submission-only)"
            )
            continue
        print(f"\n[{lang.upper()}] - {data['Total Queries']} Queries Evaluated")
        print(f"  ├─ Dense Only:    {_format_dense_sparse_metrics(lang_metrics['dense'])}")
        print(f"  ├─ Sparse Only:   {_format_dense_sparse_metrics(lang_metrics['sparse'])}")
        print(
            f"  ├─ {fusion_label}:    MRR@5: {lang_metrics['rrf']['mrr5']:.4f} | R@{config.fusion_top_k}: {lang_metrics['rrf'][f'r{config.fusion_top_k}']:.4f}"
        )
        print(
            f"  └─ Final Rerank:  MRR@5: {lang_metrics['final']['mrr5']:.4f} | R@5: {lang_metrics['final']['r5']:.4f}"
        )

    print(
        "\n================================================================================"
    )
    if summary["global"] is None:
        print(
            "[GLOBAL SUMMARY] - No labeled queries available; skipped metric "
            "aggregation for this run."
        )
        print(
            f"  Total queries: {summary['total_queries']} | "
            f"Unlabeled queries: {summary['total_unlabeled_queries']}"
        )
    else:
        print(
            f"[GLOBAL AVERAGE] - {summary['total_labeled_queries']} Total Labeled "
            "Queries Across All Languages"
        )
        print(
            f"  ├─ Overall Dense:    {_format_dense_sparse_metrics(summary['global']['dense'])}"
        )
        print(
            f"  ├─ Overall Sparse:   {_format_dense_sparse_metrics(summary['global']['sparse'])}"
        )
        print(
            f"  ├─ Overall {fusion_label}:      MRR@5: {summary['global']['rrf']['mrr5']:.4f} | R@{config.fusion_top_k}: {summary['global']['rrf'][f'r{config.fusion_top_k}']:.4f}"
        )
        print(
            f"  └─ Overall Final:    MRR@5: {summary['global']['final']['mrr5']:.4f} | R@5: {summary['global']['final']['r5']:.4f}"
        )
    print(
        "================================================================================\n"
    )
    logger.info("Evaluation completed in %.2fs", timer.elapsed_seconds())

    submission_artifacts = None
    if collect_submission:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        split_label = _submission_split_label(split, query_shard)
        mounted_output_dir = (
            f"{CACHE_MOUNT}/{submission_volume_subdir.strip('/')}/{split_label}-{run_id}"
        )
        written_paths = write_submission_tsv_files(
            submission_predictions,
            output_dir=mounted_output_dir,
        )
        embedding_cache.commit()
        remote_dir = submission_volume_remote_dir(mounted_output_dir, CACHE_MOUNT)
        submission_artifacts = {
            "volume_name": EMBEDDING_CACHE_VOLUME_NAME,
            "remote_dir": remote_dir,
            "written_files": [path.rsplit("/", 1)[-1] for path in written_paths],
            "query_shard": query_shard,
        }

    return {
        "global_results": global_results,
        "submission_predictions": submission_predictions,
        "submission_artifacts": submission_artifacts,
    }


@app.local_entrypoint()
def main(
    force_recompute_sparse_cache: bool = False,
    force_recompute_dense_documents: bool = False,
    force_recompute_dense_queries: bool = False,
    split: str = "dev",
    export_submission_tsv: bool = False,
    submission_volume_subdir: str = "submissions",
    submission_download_dir: str = "submissions",
    fusion_method: str = "rrf",
    fusion_top_k: int = 60,
    force_retrain_fusion: bool = False,
    hf_fusion_repo_id: str | None = "boyes-boys-clef-2026/random-forest-fuser",
    profile: str = "custom",
    dense_model: str = "harrier-27b",
    disable_sparse: bool = False,
    reranker_model: str = "nemotron",
    disable_reranker: bool = False,
    sparse_vanilla: bool = False,
    metrics_output_file: str | None = None,
    custom_papers: str | None = None,
    languages: str = "de,fr,en",
    rerank_parallelism: int = 1,
    rerank_chunk_size: int | None = None,
    query_shard: str | None = None,
):
    """Local CLI entrypoint that dispatches Modal evaluation and export.

    Args:
        force_recompute_sparse_cache: Recompute sparse query cache even if present.
        force_recompute_dense_documents: Recompute dense document embeddings.
        force_recompute_dense_queries: Recompute dense query embeddings.
        split: Dataset split to evaluate (``train``, ``dev``, or ``test``).
        export_submission_tsv: Whether to generate submission TSV files.
        submission_volume_subdir: Remote directory prefix in Modal volume.
        submission_download_dir: Local destination for downloaded submission files.
        hf_fusion_repo_id: Hugging Face repository ID to push/pull fusion models.
    """
    split = normalize_split(split)
    _validate_metrics_output_request(split, metrics_output_file)

    run_output = evaluate_pipeline.remote(
        force_recompute_sparse_cache=force_recompute_sparse_cache,
        force_recompute_dense_documents=force_recompute_dense_documents,
        force_recompute_dense_queries=force_recompute_dense_queries,
        split=split,
        collect_submission=export_submission_tsv,
        submission_volume_subdir=submission_volume_subdir,
        fusion_method=fusion_method,
        fusion_top_k=fusion_top_k,
        force_retrain_fusion=force_retrain_fusion,
        hf_fusion_repo_id=hf_fusion_repo_id,
        profile=profile,
        dense_model=dense_model,
        disable_sparse=disable_sparse,
        reranker_model=reranker_model,
        disable_reranker=disable_reranker,
        sparse_vanilla=sparse_vanilla,
        custom_papers=custom_papers,
        languages=[lang.strip() for lang in languages.split(",") if lang.strip()],
        rerank_parallelism=rerank_parallelism,
        rerank_chunk_size=rerank_chunk_size,
        query_shard=query_shard,
    )

    if metrics_output_file:
        import json

        with open(metrics_output_file, "w") as f:
            json.dump(run_output["global_results"], f, indent=2)
        print(f"Saved metrics to {metrics_output_file}")

    if export_submission_tsv:
        submission_artifacts = run_output["submission_artifacts"]
        download_command = modal_volume_download_command(
            volume_name=submission_artifacts["volume_name"],
            remote_dir=submission_artifacts["remote_dir"],
            local_destination=submission_download_dir,
        )
        print("Wrote submission files to Modal volume:")
        print(f"  - Volume: {submission_artifacts['volume_name']}")
        print(f"  - Remote dir: {submission_artifacts['remote_dir']}")
        print("  - Files:")
        for filename in submission_artifacts["written_files"]:
            print(f"    - {filename}")
        print("\nRun this command to download them locally:")
        print(f"  {download_command}")
