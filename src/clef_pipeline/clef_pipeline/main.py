"""Modal entrypoint for multilingual evaluation and submission export."""

import modal

from .logging_utils import StageTimer, get_logger
from .metrics import EvaluationMetrics
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
logger = get_logger("clef_pipeline.modal")


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
    print(
        f"Dense Only:   MRR@5: {metrics['dense']['mrr5']:.4f} | R@5: {metrics['dense']['r5']:.4f} | R@10: {metrics['dense']['r10']:.4f} | R@30: {metrics['dense']['r30']:.4f} | R@50: {metrics['dense']['r50']:.4f}"
    )
    print(
        f"Sparse Only:  MRR@5: {metrics['sparse']['mrr5']:.4f} | R@5: {metrics['sparse']['r5']:.4f} | R@10: {metrics['sparse']['r10']:.4f} | R@30: {metrics['sparse']['r30']:.4f} | R@50: {metrics['sparse']['r50']:.4f}"
    )
    print(
        f"{fusion_label} Output:   MRR@5: {metrics['rrf']['mrr5']:.4f} | R@{fusion_top_k}: {metrics['rrf'][f'r{fusion_top_k}']:.4f}"
    )
    print(
        f"Final Rerank: MRR@5: {metrics['final']['mrr5']:.4f} | R@5: {metrics['final']['r5']:.4f}"
    )


def _validate_metrics_output_request(split: str, metrics_output_file: str | None) -> None:
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


@app.function(
    image=image,
    gpu="H100",
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
    force_retrain_fusion: bool = False,
    hf_fusion_repo_id: str | None = "boyes-boys-clef-2026/random-forest-fuser",
    profile: str = "custom",
    dense_model: str = "harrier-27b",
    disable_sparse: bool = False,
    reranker_model: str = "nemotron",
    disable_reranker: bool = False,
    sparse_vanilla: bool = False,
    custom_papers: str | None = None,
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
    import os
    from datetime import datetime, timezone

    from datasets import load_dataset
    from tqdm import tqdm

    timer = StageTimer()
    split = normalize_split(split)

    sparse_k1 = 1.5 if sparse_vanilla else 2.5
    sparse_b = 0.75 if sparse_vanilla else 0.85
    sparse_use_bigrams = not sparse_vanilla
    sparse_use_translation = not sparse_vanilla

    config = build_pipeline_config(
        profile,
        fusion_method=fusion_method,
        hf_fusion_repo_id=hf_fusion_repo_id,
        hf_token=os.environ.get("HF_TOKEN"),
        dense_model=dense_model,
        disable_sparse=disable_sparse,
        reranker_model=reranker_model,
        disable_reranker=disable_reranker,
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
    if custom_papers:
        custom_path = os.path.join(CACHE_MOUNT, custom_papers)
        logger.info(f"Appending custom papers from: {custom_path}")
        custom_docs = read_custom_papers(custom_path, 11000)
        collection_documents.extend(custom_docs)
        logger.info(f"Total documents after append: {len(collection_documents)}")
    pipeline.index_collection(
        collection_documents=collection_documents,
        cache_dir=CACHE_MOUNT,
        force_recompute_dense_documents=force_recompute_dense_documents,
    )
    embedding_cache.commit()

    languages = ["de", "fr", "en"]
    lang_tweets: dict[str, list[dict]] = {}
    cache_langs = {lang: f"{split}_{lang}" for lang in languages}
    for lang in languages:
        tweets = load_query_split(lang, split)
        lang_tweets[lang] = tweets
        query_texts = [row["text"] for row in tweets]
        pipeline.index_queries_for_language(
            lang=lang,
            query_texts=query_texts,
            cache_lang=cache_langs[lang],
            cache_dir=CACHE_MOUNT,
            force_recompute_sparse_cache=force_recompute_sparse_cache,
            force_recompute_dense_queries=force_recompute_dense_queries,
        )
        embedding_cache.commit()

    pipeline.prepare_fusion_model(
        cache_dir=CACHE_MOUNT,
        languages=languages,
        force_retrain_fusion=force_retrain_fusion,
        force_recompute_dense_queries=force_recompute_dense_queries,
        force_recompute_sparse_cache=force_recompute_sparse_cache,
        on_cache_update=embedding_cache.commit,
    )

    pipeline.unload_dense_models()

    logger.info("Running multilingual evaluation...")
    fusion_label = "RF Fusion" if config.fusion_method == "random_forest" else "RRF"
    metrics = EvaluationMetrics(fusion_top_k=config.fusion_top_k)
    submission_predictions = (
        {lang: [] for lang in languages} if collect_submission else {}
    )

    for lang in languages:
        tweets = lang_tweets[lang]
        print("\n==========================================")
        print(f"  STARTING EVALUATION FOR LANGUAGE: {lang.upper()}")
        print("==========================================")
        for i, row in enumerate(
            tqdm(tweets, desc=f"Evaluating {lang.upper()} Queries")
        ):
            result = pipeline.search_cached_query(
                query_idx=i,
                query_text=row["text"],
                lang=lang,
                cache_lang=cache_langs[lang],
            )

            _record_query_result(
                lang=lang,
                row=row,
                query_idx=i,
                result=result,
                metrics=metrics,
                submission_predictions=(
                    submission_predictions if collect_submission else None
                ),
            )

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
        print(
            f"  ├─ Dense Only:    MRR@5: {lang_metrics['dense']['mrr5']:.4f} | R@5: {lang_metrics['dense']['r5']:.4f} | R@10: {lang_metrics['dense']['r10']:.4f} | R@30: {lang_metrics['dense']['r30']:.4f} | R@50: {lang_metrics['dense']['r50']:.4f}"
        )
        print(
            f"  ├─ Sparse Only:   MRR@5: {lang_metrics['sparse']['mrr5']:.4f} | R@5: {lang_metrics['sparse']['r5']:.4f} | R@10: {lang_metrics['sparse']['r10']:.4f} | R@30: {lang_metrics['sparse']['r30']:.4f} | R@50: {lang_metrics['sparse']['r50']:.4f}"
        )
        print(
            f"  ├─ {fusion_label}:    MRR@5: {lang_metrics['rrf']['mrr5']:.4f} | R@{config.fusion_top_k}: {lang_metrics['rrf'][f'r{config.fusion_top_k}']:.4f}"
        )
        print(
            f"  └─ Final Rerank:  MRR@5: {lang_metrics['final']['mrr5']:.4f} | R@5: {lang_metrics['final']['r5']:.4f}"
        )

    print("\n================================================================================")
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
            f"  ├─ Overall Dense:    MRR@5: {summary['global']['dense']['mrr5']:.4f} | R@5: {summary['global']['dense']['r5']:.4f} | R@10: {summary['global']['dense']['r10']:.4f} | R@30: {summary['global']['dense']['r30']:.4f} | R@50: {summary['global']['dense']['r50']:.4f}"
        )
        print(
            f"  ├─ Overall Sparse:   MRR@5: {summary['global']['sparse']['mrr5']:.4f} | R@5: {summary['global']['sparse']['r5']:.4f} | R@10: {summary['global']['sparse']['r10']:.4f} | R@30: {summary['global']['sparse']['r30']:.4f} | R@50: {summary['global']['sparse']['r50']:.4f}"
        )
        print(
            f"  ├─ Overall {fusion_label}:      MRR@5: {summary['global']['rrf']['mrr5']:.4f} | R@{config.fusion_top_k}: {summary['global']['rrf'][f'r{config.fusion_top_k}']:.4f}"
        )
        print(
            f"  └─ Overall Final:    MRR@5: {summary['global']['final']['mrr5']:.4f} | R@5: {summary['global']['final']['r5']:.4f}"
        )
    print("================================================================================\n")
    logger.info("Evaluation completed in %.2fs", timer.elapsed_seconds())

    submission_artifacts = None
    if collect_submission:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        mounted_output_dir = (
            f"{CACHE_MOUNT}/{submission_volume_subdir.strip('/')}/{split}-{run_id}"
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
        force_retrain_fusion=force_retrain_fusion,
        hf_fusion_repo_id=hf_fusion_repo_id,
        profile=profile,
        dense_model=dense_model,
        disable_sparse=disable_sparse,
        reranker_model=reranker_model,
        disable_reranker=disable_reranker,
        sparse_vanilla=sparse_vanilla,
        custom_papers=custom_papers,
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
