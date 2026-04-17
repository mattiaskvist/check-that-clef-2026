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
from .utils import CHECKTHAT_DATASET

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "sentence-transformers",
        "peft",
        "rank_bm25",
        "datasets",
        "tqdm",
        "transformers",
        "accelerate",
        "nltk",
        "Pillow",
        "torchvision",
        "deep-translator",
    )
)

app = modal.App("checkthat-evaluation-pipeline")
EMBEDDING_CACHE_VOLUME_NAME = "checkthat-embedding-cache"
embedding_cache = modal.Volume.from_name(
    EMBEDDING_CACHE_VOLUME_NAME, create_if_missing=True
)
CACHE_MOUNT = "/cache/embeddings"
logger = get_logger("clef_pipeline.modal")


def _print_language_summary(lang: str, data: dict[str, object], fusion_top_k: int):
    metrics = data["metrics"]
    if metrics is None:
        print(f"\n--- Summary for {lang.upper()} ({data['Total Queries']} Queries) ---")
        print("No labels available; generated predictions only.")
        return

    print(f"\n--- Summary for {lang.upper()} ({data['Total Queries']} Queries) ---")
    print(
        f"Dense Only:   MRR@5: {metrics['dense']['mrr5']:.4f} | R@5: {metrics['dense']['r5']:.4f} | R@10: {metrics['dense']['r10']:.4f} | R@30: {metrics['dense']['r30']:.4f} | R@50: {metrics['dense']['r50']:.4f}"
    )
    print(
        f"Sparse Only:  MRR@5: {metrics['sparse']['mrr5']:.4f} | R@5: {metrics['sparse']['r5']:.4f} | R@10: {metrics['sparse']['r10']:.4f} | R@30: {metrics['sparse']['r30']:.4f} | R@50: {metrics['sparse']['r50']:.4f}"
    )
    print(
        f"RRF Output:   MRR@5: {metrics['rrf']['mrr5']:.4f} | R@{fusion_top_k}: {metrics['rrf'][f'r{fusion_top_k}']:.4f}"
    )
    print(
        f"Final Rerank: MRR@5: {metrics['final']['mrr5']:.4f} | R@5: {metrics['final']['r5']:.4f}"
    )


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 3,
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
):
    from datetime import datetime, timezone

    from datasets import load_dataset
    from tqdm import tqdm

    timer = StageTimer()
    split = normalize_split(split)
    config = build_pipeline_config("evaluation")
    pipeline = build_pipeline_from_config(config)

    logger.info("Loading collection and building index...")
    collection_dataset = load_dataset(
        CHECKTHAT_DATASET, "collection", split="collection"
    )
    collection_documents = collection_dataset.to_list()
    pipeline.index_collection(
        collection_documents=collection_documents,
        cache_dir=CACHE_MOUNT,
        force_recompute_dense_documents=force_recompute_dense_documents,
    )
    embedding_cache.commit()

    languages = ["de", "fr", "en"]
    lang_tweets: dict[str, list[dict]] = {}
    for lang in languages:
        tweets = list(load_dataset(CHECKTHAT_DATASET, lang)[split])
        lang_tweets[lang] = tweets
        query_texts = [row["text"] for row in tweets]
        pipeline.index_queries_for_language(
            lang=lang,
            query_texts=query_texts,
            cache_dir=CACHE_MOUNT,
            force_recompute_sparse_cache=force_recompute_sparse_cache,
            force_recompute_dense_queries=force_recompute_dense_queries,
        )
        embedding_cache.commit()

    pipeline.unload_dense_models()

    logger.info("Running multilingual evaluation...")
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
            )

            if collect_submission:
                submission_predictions[lang].append(
                    {
                        "index": row.get("index", i),
                        "preds": result["preds"][:SUBMISSION_TOP_K],
                    }
                )

            metrics.add_query(
                lang=lang,
                true_pubkey=row.get("pubkey"),
                stages=result["stages"],
            )

    summary = metrics.summary()
    global_results = summary["languages"]
    for lang in languages:
        _print_language_summary(lang, global_results[lang], config.fusion_top_k)

    print("\n\n" + "*" * 80)
    print("*" + " FINAL MULTILINGUAL EVALUATION SUMMARY ".center(78) + "*")
    print("*" * 80)
    for lang in languages:
        data = global_results[lang]
        lang_metrics = data["metrics"]
        if lang_metrics is None:
            print(f"\n[{lang.upper()}] - {data['Total Queries']} Queries (No labels)")
            continue
        print(f"\n[{lang.upper()}] - {data['Total Queries']} Queries Evaluated")
        print(
            f"  ├─ Dense Only:    MRR@5: {lang_metrics['dense']['mrr5']:.4f} | R@5: {lang_metrics['dense']['r5']:.4f} | R@10: {lang_metrics['dense']['r10']:.4f} | R@30: {lang_metrics['dense']['r30']:.4f} | R@50: {lang_metrics['dense']['r50']:.4f}"
        )
        print(
            f"  ├─ Sparse Only:   MRR@5: {lang_metrics['sparse']['mrr5']:.4f} | R@5: {lang_metrics['sparse']['r5']:.4f} | R@10: {lang_metrics['sparse']['r10']:.4f} | R@30: {lang_metrics['sparse']['r30']:.4f} | R@50: {lang_metrics['sparse']['r50']:.4f}"
        )
        print(
            f"  ├─ RRF Output:    MRR@5: {lang_metrics['rrf']['mrr5']:.4f} | R@{config.fusion_top_k}: {lang_metrics['rrf'][f'r{config.fusion_top_k}']:.4f}"
        )
        print(
            f"  └─ Final Rerank:  MRR@5: {lang_metrics['final']['mrr5']:.4f} | R@5: {lang_metrics['final']['r5']:.4f}"
        )

    print(
        "\n================================================================================"
    )
    print(
        f"[GLOBAL AVERAGE] - {summary['total_labeled_queries']} Total Queries Across All Languages"
    )
    print(
        f"  ├─ Overall Dense:    MRR@5: {summary['global']['dense']['mrr5']:.4f} | R@5: {summary['global']['dense']['r5']:.4f} | R@10: {summary['global']['dense']['r10']:.4f} | R@30: {summary['global']['dense']['r30']:.4f} | R@50: {summary['global']['dense']['r50']:.4f}"
    )
    print(
        f"  ├─ Overall Sparse:   MRR@5: {summary['global']['sparse']['mrr5']:.4f} | R@5: {summary['global']['sparse']['r5']:.4f} | R@10: {summary['global']['sparse']['r10']:.4f} | R@30: {summary['global']['sparse']['r30']:.4f} | R@50: {summary['global']['sparse']['r50']:.4f}"
    )
    print(
        f"  ├─ Overall RRF:      MRR@5: {summary['global']['rrf']['mrr5']:.4f} | R@{config.fusion_top_k}: {summary['global']['rrf'][f'r{config.fusion_top_k}']:.4f}"
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
):
    run_output = evaluate_pipeline.remote(
        force_recompute_sparse_cache=force_recompute_sparse_cache,
        force_recompute_dense_documents=force_recompute_dense_documents,
        force_recompute_dense_queries=force_recompute_dense_queries,
        split=split,
        collect_submission=export_submission_tsv,
        submission_volume_subdir=submission_volume_subdir,
    )

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
