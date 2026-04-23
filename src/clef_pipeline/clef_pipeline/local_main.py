"""Local (non-Modal) entrypoint for multilingual evaluation and submission export."""

from __future__ import annotations

import argparse
import json
import os

from .logging_utils import StageTimer, get_logger
from .metrics import EvaluationMetrics
from .pipeline_config import build_pipeline_config
from .registry import build_pipeline_from_config
from .submission import (
    SUBMISSION_TOP_K,
    normalize_split,
    write_submission_tsv_files,
)
from .utils import CHECKTHAT_DATASET

logger = get_logger("clef_pipeline.local")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="clef_pipeline.local",
        description="Run the CheckThat CLEF retrieval pipeline locally (no Modal).",
    )
    parser.add_argument("--profile", default="custom")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--cache-dir", default=os.path.join(".cache", "embeddings"))

    parser.add_argument("--fusion-method", default="rrf", choices=["rrf", "random_forest"])
    parser.add_argument("--dense-model", default="harrier-27b")
    parser.add_argument("--disable-sparse", action="store_true")
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--reranker-model", default="nemotron")

    parser.add_argument("--sparse-vanilla", action="store_true")
    parser.add_argument("--force-recompute-sparse-cache", action="store_true")
    parser.add_argument("--force-recompute-dense-documents", action="store_true")
    parser.add_argument("--force-recompute-dense-queries", action="store_true")
    parser.add_argument("--force-retrain-fusion", action="store_true")
    parser.add_argument("--hf-fusion-repo-id", default=None)

    parser.add_argument("--export-submission-tsv", action="store_true")
    parser.add_argument("--submission-dir", default="submissions")
    parser.add_argument("--metrics-output-file", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from datasets import load_dataset
    from tqdm import tqdm

    args = _parse_args(argv)
    timer = StageTimer()
    split = normalize_split(args.split)

    sparse_k1 = 1.5 if args.sparse_vanilla else 2.5
    sparse_b = 0.75 if args.sparse_vanilla else 0.85
    sparse_use_bigrams = not args.sparse_vanilla
    sparse_use_translation = not args.sparse_vanilla

    config = build_pipeline_config(
        args.profile,
        fusion_method=args.fusion_method,
        hf_fusion_repo_id=args.hf_fusion_repo_id,
        hf_token=os.environ.get("HF_TOKEN"),
        dense_model=args.dense_model,
        disable_sparse=args.disable_sparse,
        reranker_model=args.reranker_model,
        disable_reranker=args.disable_reranker,
        sparse_k1=sparse_k1,
        sparse_b=sparse_b,
        sparse_use_bigrams=sparse_use_bigrams,
        sparse_use_translation=sparse_use_translation,
    )
    pipeline = build_pipeline_from_config(config)

    logger.info("Loading collection and building index...")
    collection_dataset = load_dataset(CHECKTHAT_DATASET, "collection", split="collection")
    collection_documents = collection_dataset.to_list()
    pipeline.index_collection(
        collection_documents=collection_documents,
        cache_dir=args.cache_dir,
        force_recompute_dense_documents=args.force_recompute_dense_documents,
    )

    languages = ["de", "fr", "en"]
    lang_tweets: dict[str, list[dict]] = {}
    for lang in languages:
        tweets = list(load_dataset(CHECKTHAT_DATASET, lang)[split])
        lang_tweets[lang] = tweets
        query_texts = [row["text"] for row in tweets]
        pipeline.index_queries_for_language(
            lang=lang,
            query_texts=query_texts,
            cache_dir=args.cache_dir,
            force_recompute_sparse_cache=args.force_recompute_sparse_cache,
            force_recompute_dense_queries=args.force_recompute_dense_queries,
        )

    pipeline.prepare_fusion_model(
        cache_dir=args.cache_dir,
        languages=languages,
        force_retrain_fusion=args.force_retrain_fusion,
        force_recompute_dense_queries=args.force_recompute_dense_queries,
        force_recompute_sparse_cache=args.force_recompute_sparse_cache,
        on_cache_update=None,
    )

    pipeline.unload_dense_models()

    logger.info("Running multilingual evaluation...")
    metrics = EvaluationMetrics(fusion_top_k=config.fusion_top_k)
    submission_predictions = (
        {lang: [] for lang in languages} if args.export_submission_tsv else {}
    )

    for lang in languages:
        tweets = lang_tweets[lang]
        for i, row in enumerate(tqdm(tweets, desc=f"Evaluating {lang.upper()} Queries")):
            result = pipeline.search_cached_query(
                query_idx=i,
                query_text=row["text"],
                lang=lang,
            )

            if args.export_submission_tsv:
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
    _ = timer.elapsed_seconds()

    if args.metrics_output_file:
        with open(args.metrics_output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved metrics to {args.metrics_output_file}")

    if args.export_submission_tsv:
        os.makedirs(args.submission_dir, exist_ok=True)
        written = write_submission_tsv_files(
            predictions_by_lang=submission_predictions,
            output_dir=args.submission_dir,
        )
        print(f"Wrote submission files to {args.submission_dir}:")
        for filename in written:
            print(f"  - {filename}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
