"""Modal entrypoint for multilingual evaluation and submission export."""

import modal

from .fusions import RandomForestFuser
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
    # Reduce fragmentation during Harrier-27B encoding: the sliding-window
    # causal mask is allocated as a big contiguous tensor and fails on a
    # fragmented pool even when total free memory is sufficient.
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        # Pin <5: sentence-transformers 5.x calls AutoProcessor inside the
        # Transformer module, which tries to load an image processor for
        # multimodal-flagged HF repos like microsoft/harrier-oss-v1-27b
        # and crashes. v4.x uses AutoTokenizer and works for our text-only
        # retrievers (Harrier, BGE-M3, e5-large, jina-v3).
        "sentence-transformers>=3.0,<5",
        "peft",
        "rank_bm25",
        "datasets",
        "tqdm",
        # Pin <4.55: transformers 4.55+ introduced a vmap-based sliding-window
        # mask builder (sdpa_mask_recent_torch, PR #41265) that materializes
        # a 5-level-nested per-head mask and causes ~5 GiB allocations during
        # Harrier-27B inference, OOMing on A100-80GB. The pre-4.55 builder
        # is the memory-efficient path we were implicitly using before.
        "transformers>=4.41,<4.55",
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
# Shared data volume holding training artifacts. Mounted read-mostly so the
# evaluation pipeline can load a local reranker checkpoint by passing
# ``reranker_model_name=/data/<checkpoint-dir>``.
CLEF_VOLUME_NAME = "clef-vol"
clef_volume = modal.Volume.from_name(CLEF_VOLUME_NAME, create_if_missing=True)
DATA_MOUNT = "/data"
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
        f"{fusion_label} Output:   MRR@5: {metrics['rrf']['mrr5']:.4f} | R@{fusion_top_k}: {metrics['rrf'][f'r{fusion_top_k}']:.4f}"
    )
    print(
        f"Final Rerank: MRR@5: {metrics['final']['mrr5']:.4f} | R@5: {metrics['final']['r5']:.4f}"
    )


@app.function(
    image=image,
    # B200 (Blackwell, 192 GB HBM, new-generation drivers) handles every
    # model in the stack — Harrier-27B encoding, Qwen3-Reranker-8B, Gemma
    # rerankers — without the driver-mismatch failure seen on A100-80GB.
    # Plenty of room to run large rerankers at bigger micro batches too.
    gpu="H100",
    # Budget: worst case is Qwen3-Reranker-8B on EN alone with
    # fusion_top_k=100. Measured ~10 s/query on B200 => ~11 h for 3905
    # queries, plus model cold start (~60 s) and Harrier query encode
    # (~15 min if cache miss). 15 h gives comfortable headroom.
    timeout=60 * 60 * 15,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={CACHE_MOUNT: embedding_cache, DATA_MOUNT: clef_volume},
)
def evaluate_pipeline(
    force_recompute_sparse_cache: bool = False,
    force_recompute_dense_documents: bool = False,
    force_recompute_dense_queries: bool = False,
    split: str = "dev",
    collect_submission: bool = False,
    submission_volume_subdir: str = "submissions",
    fusion_method: str = "rrf",
    fusion_weight_dense: float = 0.8,
    fusion_weight_sparse: float = 0.2,
    fusion_top_k: int = 100,
    force_retrain_fusion: bool = False,
    global_fusion_model: bool = False,
    reranker_name: str | None = None,
    reranker_model_name: str | None = None,
    languages_csv: str | None = None,
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
        fusion_weight_dense: Weight on the dense retriever in weighted RRF.
        fusion_weight_sparse: Weight on the sparse retriever in weighted RRF.
        fusion_top_k: Candidate pool size handed to the reranker.
        force_retrain_fusion: Force retraining learned fusion model instead of loading cache.
        global_fusion_model: Train one fusion model for all languages.
        reranker_name: Optional reranker registry name to override the
            evaluation profile default (``"qwen3-reranker-8b"``). Pass the
            empty string to disable reranking entirely.
        reranker_model_name: Optional ``model_name`` forwarded to the
            reranker constructor (e.g. a local checkpoint path on the
            ``clef-vol`` volume).
        languages_csv: Comma-separated subset of language codes to evaluate
            (e.g. ``"de"`` or ``"de,fr"``). If empty/None, all supported
            languages (``de``, ``fr``, ``en``) are evaluated. Useful for
            GPU-budget-constrained runs that only need a single language —
            DE reranking with 386 queries costs ~18 min vs. ~4h for all three.

    Returns:
        Global language results and optional submission artifact metadata.
    """
    import json
    import os
    from datetime import datetime, timezone

    from datasets import load_dataset
    from tqdm import tqdm

    timer = StageTimer()
    split = normalize_split(split)
    fusion_weights = (float(fusion_weight_dense), float(fusion_weight_sparse))
    reranker_params = (
        {"model_name": reranker_model_name} if reranker_model_name else None
    )
    config = build_pipeline_config(
        "evaluation",
        fusion_method=fusion_method,
        fusion_weights=fusion_weights,
        fusion_top_k=fusion_top_k,
        reranker_name=reranker_name,
        reranker_params=reranker_params,
    )
    pipeline = build_pipeline_from_config(config)
    if config.fusion_method == "random_forest":
        pipeline.fuser = RandomForestFuser(global_model=global_fusion_model)

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

    ALL_LANGUAGES = ["de", "fr", "en"]
    if languages_csv:
        requested = [code.strip().lower() for code in languages_csv.split(",") if code.strip()]
        unknown = [code for code in requested if code not in ALL_LANGUAGES]
        if unknown:
            raise ValueError(
                f"Unknown language codes in languages_csv={languages_csv!r}: "
                f"{unknown}. Supported: {ALL_LANGUAGES}"
            )
        # Preserve the canonical DE → FR → EN order so cache lookups and log
        # output stay deterministic regardless of the order given on the CLI.
        languages = [code for code in ALL_LANGUAGES if code in requested]
        print(f"[languages] Evaluating subset: {languages}")
    else:
        languages = ALL_LANGUAGES
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

    if config.fusion_method == "random_forest":
        dense_retriever, sparse_retriever = pipeline.get_fusion_retrievers()
        if not isinstance(pipeline.fuser, RandomForestFuser):
            raise RuntimeError("Expected RandomForestFuser for random_forest mode.")

        sparse_config = {
            "k1": getattr(sparse_retriever, "bm25_k1", None),
            "b": getattr(sparse_retriever, "bm25_b", None),
            "stemmer": "lancaster",
        }
        dense_model_name = getattr(
            dense_retriever, "model_name", dense_retriever.__class__.__name__
        )

        loaded = False
        if not force_retrain_fusion:
            loaded = pipeline.fuser.load(
                cache_dir=CACHE_MOUNT,
                dense_model_name=dense_model_name,
                sparse_config=sparse_config,
                train_split="train",
            )

        if not loaded:
            train_tweets_by_lang: dict[str, list[dict]] = {}
            for lang in languages:
                train_tweets = list(load_dataset(CHECKTHAT_DATASET, lang)["train"])
                train_tweets_by_lang[lang] = train_tweets
                train_query_texts = [row["text"] for row in train_tweets]

                dense_retriever.index_queries(
                    train_query_texts,
                    cache_dir=CACHE_MOUNT,
                    cache_name=f"queries_train_{lang}",
                    force_recompute=force_recompute_dense_queries,
                )
                sparse_retriever.index_queries(
                    train_query_texts,
                    lang=lang,
                    cache_dir=CACHE_MOUNT,
                    cache_name=f"sparse_queries_train_{lang}",
                    top_k=config.sparse_cache_top_k,
                    force_recompute=force_recompute_sparse_cache,
                )
                embedding_cache.commit()

            pipeline.fuser.train(
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
                train_tweets_by_lang=train_tweets_by_lang,
                article_pubkeys=pipeline.article_pubkeys,
                dense_model_name=dense_model_name,
                sparse_config=sparse_config,
                train_split="train",
            )
            pipeline.fuser.save(cache_dir=CACHE_MOUNT)
            embedding_cache.commit()

    pipeline.unload_dense_models()

    logger.info("Running multilingual evaluation...")
    fusion_label_map = {
        "random_forest": "RF Fusion",
        "rrf": "RRF",
    }
    fusion_label = fusion_label_map.get(config.fusion_method, config.fusion_method)
    metrics = EvaluationMetrics(fusion_top_k=config.fusion_top_k)
    submission_predictions = (
        {lang: [] for lang in languages} if collect_submission else {}
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Partial results checkpoint directory: per-language metrics are flushed
    # here as each language completes so a container timeout (or crash) still
    # leaves usable numbers behind. This was added after a run hit the
    # function timeout mid-EN reranking and lost every metric — DE and FR had
    # already been computed but never printed because the aggregation is at
    # the end of the language loop.
    partial_results_dir = (
        f"{CACHE_MOUNT}/partial_results/{split}-{run_id}"
    )
    os.makedirs(partial_results_dir, exist_ok=True)

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

        # Language complete: print metrics now and flush to volume so a
        # timeout during a later language doesn't erase this one's work.
        try:
            partial_summary = metrics.summary()
            lang_data = partial_summary["languages"][lang]
            _print_language_summary(
                lang, lang_data, config.fusion_top_k, fusion_label
            )
            partial_path = os.path.join(
                partial_results_dir, f"metrics_{lang}.json"
            )
            with open(partial_path, "w", encoding="utf-8") as handle:
                json.dump(lang_data, handle, ensure_ascii=False, indent=2)
            print(f"[checkpoint] Wrote {lang.upper()} metrics to {partial_path}")
        except Exception as exc:  # noqa: BLE001 — best-effort checkpoint
            print(f"[checkpoint] Failed to persist {lang.upper()} metrics: {exc}")

        # Commit the cache volume so partial metrics survive.
        try:
            embedding_cache.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[checkpoint] Volume commit failed for {lang.upper()}: {exc}")

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
            f"  ├─ {fusion_label}:    MRR@5: {lang_metrics['rrf']['mrr5']:.4f} | R@{config.fusion_top_k}: {lang_metrics['rrf'][f'r{config.fusion_top_k}']:.4f}"
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
        "run_id": run_id,
        "config": {
            "fusion_method": config.fusion_method,
            "fusion_weights": list(config.fusion_weights)
            if config.fusion_weights is not None
            else None,
            "fusion_top_k": config.fusion_top_k,
            "final_top_k": config.final_top_k,
        },
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
    fusion_weight_dense: float = 0.8,
    fusion_weight_sparse: float = 0.2,
    fusion_top_k: int = 100,
    force_retrain_fusion: bool = False,
    global_fusion_model: bool = False,
    reranker_name: str | None = None,
    reranker_model_name: str | None = None,
    languages_csv: str | None = None,
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
        fusion_method: Fusion strategy (``rrf`` or ``random_forest``).
        fusion_weight_dense: Dense retriever weight in weighted RRF.
        fusion_weight_sparse: Sparse retriever weight in weighted RRF.
        fusion_top_k: Candidate pool size handed to the reranker.
        reranker_name: Optional reranker registry name override (e.g.
            ``qwen3-reranker-8b``). Empty string disables reranking.
        reranker_model_name: Optional model_name/path forwarded to the
            reranker constructor. Use a ``/data/...`` path on ``clef-vol`` to
            load a locally cached checkpoint.
        languages_csv: Comma-separated subset of language codes to evaluate
            (``"de"``, ``"de,fr"``, etc.). Empty/None evaluates all three.
    """
    run_output = evaluate_pipeline.remote(
        force_recompute_sparse_cache=force_recompute_sparse_cache,
        force_recompute_dense_documents=force_recompute_dense_documents,
        force_recompute_dense_queries=force_recompute_dense_queries,
        split=split,
        collect_submission=export_submission_tsv,
        submission_volume_subdir=submission_volume_subdir,
        fusion_method=fusion_method,
        fusion_weight_dense=fusion_weight_dense,
        fusion_weight_sparse=fusion_weight_sparse,
        fusion_top_k=fusion_top_k,
        force_retrain_fusion=force_retrain_fusion,
        global_fusion_model=global_fusion_model,
        reranker_name=reranker_name,
        reranker_model_name=reranker_model_name,
        languages_csv=languages_csv,
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

