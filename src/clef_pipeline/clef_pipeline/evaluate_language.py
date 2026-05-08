"""Provider-neutral evaluation runner for GPU VMs and Vertex AI.

The Modal entrypoint remains in ``clef_pipeline.main``. This module runs the
same retrieval/evaluation path without Modal so a single Google Cloud machine
with multiple visible GPUs can evaluate one or more languages in parallel.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import DENSE_SPARSE_RECALL_CUTOFFS, EvaluationMetrics
from .pipeline_config import build_pipeline_config
from .registry import build_pipeline_from_config
from .submission import (
    SUBMISSION_TOP_K,
    normalize_split,
    write_submission_tsv_files,
)
from .utils import CHECKTHAT_DATASET, load_query_split, read_custom_papers


DEFAULT_CACHE_DIR = "/data/checkthat-eval-cache"
DEFAULT_WORK_DIR = "/data/checkthat-eval-work"
DEFAULT_LANGUAGES = "de"
DEFAULT_GPU_IDS = "0,1,2,3"


def _format_dense_sparse_metrics(stage_metrics: dict[str, float]) -> str:
    """Format MRR and recall metrics for dense/sparse stages."""
    recall_metrics = " | ".join(
        f"R@{cutoff}: {stage_metrics[f'r{cutoff}']:.4f}"
        for cutoff in DENSE_SPARSE_RECALL_CUTOFFS
    )
    return f"MRR@5: {stage_metrics['mrr5']:.4f} | {recall_metrics}"


def _print_language_summary(
    lang: str,
    data: dict[str, object],
    fusion_top_k: int,
    fusion_label: str,
) -> None:
    """Print one language summary."""
    metrics = data["metrics"]
    if metrics is None:
        print(f"\n--- Summary for {lang.upper()} ({data['Total Queries']} Queries) ---")
        print(
            "No labels available; generated predictions only. "
            f"Labeled: {data['Labeled Queries']}, "
            f"Unlabeled: {data['Unlabeled Queries']}."
        )
        return

    print(f"\n--- Summary for {lang.upper()} ({data['Total Queries']} Queries) ---")
    print(f"Dense Only:   {_format_dense_sparse_metrics(metrics['dense'])}")
    print(f"Sparse Only:  {_format_dense_sparse_metrics(metrics['sparse'])}")
    print(
        f"{fusion_label}:   MRR@5: {metrics['rrf']['mrr5']:.4f} | "
        f"R@{fusion_top_k}: {metrics['rrf'][f'r{fusion_top_k}']:.4f}"
    )
    print(
        f"Final Rerank: MRR@5: {metrics['final']['mrr5']:.4f} | "
        f"R@5: {metrics['final']['r5']:.4f}"
    )


def _chunk_indices(total: int, chunk_size: int) -> list[tuple[int, int]]:
    """Split a query range into half-open chunks."""
    effective_chunk_size = max(1, int(chunk_size))
    return [
        (start, min(start + effective_chunk_size, total))
        for start in range(0, total, effective_chunk_size)
    ]


def _split_indices(total: int, num_splits: int) -> list[list[int]]:
    """Split query indices into balanced groups."""
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


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_config(payload: dict[str, Any]):
    sparse_vanilla = payload["sparse_vanilla"]
    sparse_k1 = 1.5 if sparse_vanilla else 2.5
    sparse_b = 0.75 if sparse_vanilla else 0.85
    sparse_use_bigrams = not sparse_vanilla
    sparse_use_translation = not sparse_vanilla
    return build_pipeline_config(
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


def _load_collection_documents(cache_dir: str, custom_papers: str | None) -> list[dict]:
    from datasets import load_dataset

    collection_dataset = load_dataset(
        CHECKTHAT_DATASET,
        "collection",
        split="collection",
    )
    collection_documents = collection_dataset.to_list()
    if custom_papers:
        custom_path = os.path.join(cache_dir, custom_papers)
        collection_documents.extend(read_custom_papers(custom_path, 11000))
    return collection_documents


def _index_pipeline_for_languages(
    *,
    payload: dict[str, Any],
    languages: list[str],
    split: str,
    force_recompute_sparse_cache: bool,
    force_recompute_dense_documents: bool,
    force_recompute_dense_queries: bool,
    force_retrain_fusion: bool,
):
    """Build the pipeline and prepare caches for one or more languages."""
    cache_dir = payload["cache_dir"]
    config = _build_config(payload)
    pipeline = build_pipeline_from_config(config)
    collection_documents = _load_collection_documents(
        cache_dir, payload["custom_papers"]
    )

    print("Indexing collection...")
    pipeline.index_collection(
        collection_documents=collection_documents,
        cache_dir=cache_dir,
        force_recompute_dense_documents=force_recompute_dense_documents,
    )

    for lang in languages:
        tweets = load_query_split(lang, split)
        if payload.get("max_queries"):
            tweets = tweets[: int(payload["max_queries"])]
        print(f"Indexing {lang.upper()} {split} query cache ({len(tweets)} queries)...")
        pipeline.index_queries_for_language(
            lang=lang,
            query_texts=[row["text"] for row in tweets],
            cache_lang=f"{split}_{lang}",
            cache_dir=cache_dir,
            force_recompute_sparse_cache=force_recompute_sparse_cache,
            force_recompute_dense_queries=force_recompute_dense_queries,
        )

    pipeline.prepare_fusion_model(
        cache_dir=cache_dir,
        languages=languages,
        force_retrain_fusion=force_retrain_fusion,
        force_recompute_dense_queries=force_recompute_dense_queries,
        force_recompute_sparse_cache=force_recompute_sparse_cache,
    )
    pipeline.unload_dense_models()
    return pipeline


def prepare_evaluation_cache(payload: dict[str, Any]) -> None:
    """Prepare shared document/query/fusion caches before parallel workers run."""
    split = normalize_split(payload["split"])
    languages = payload["languages"]
    Path(payload["cache_dir"]).mkdir(parents=True, exist_ok=True)
    _index_pipeline_for_languages(
        payload=payload,
        languages=languages,
        split=split,
        force_recompute_sparse_cache=payload["force_recompute_sparse_cache"],
        force_recompute_dense_documents=payload["force_recompute_dense_documents"],
        force_recompute_dense_queries=payload["force_recompute_dense_queries"],
        force_retrain_fusion=payload["force_retrain_fusion"],
    )


def evaluate_worker_payload(payload_path: str) -> str:
    """Run one evaluation chunk and write JSONL results."""
    with open(payload_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    lang = payload["lang"]
    split = normalize_split(payload["split"])
    output_path = payload["output_path"]
    query_indices = payload["query_indices"]

    print(
        f"Worker evaluating {lang.upper()} {split}: "
        f"{len(query_indices)} queries -> {output_path}"
    )
    pipeline = _index_pipeline_for_languages(
        payload=payload,
        languages=[lang],
        split=split,
        force_recompute_sparse_cache=False,
        force_recompute_dense_documents=False,
        force_recompute_dense_queries=False,
        force_retrain_fusion=False,
    )

    tweets = load_query_split(lang, split)
    if payload.get("max_queries"):
        tweets = tweets[: int(payload["max_queries"])]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for query_idx in query_indices:
            row = tweets[query_idx]
            result = pipeline.search_cached_query(
                query_idx=query_idx,
                query_text=row["text"],
                lang=lang,
                cache_lang=f"{split}_{lang}",
            )
            item = {
                "lang": lang,
                "query_idx": query_idx,
                "row": row,
                "result": result,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Worker finished {output_path}")
    return output_path


def _record_query_result(
    *,
    item: dict[str, Any],
    metrics: EvaluationMetrics,
    submission_predictions: dict[str, list[dict[str, object]]] | None = None,
) -> None:
    lang = item["lang"]
    row = item["row"]
    query_idx = item["query_idx"]
    result = item["result"]
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


def _aggregate_results(
    *,
    result_paths: list[str],
    languages: list[str],
    fusion_top_k: int,
    export_submission_tsv: bool,
    submission_output_dir: str,
    metrics_output_file: str | None,
) -> dict[str, Any]:
    metrics = EvaluationMetrics(fusion_top_k=fusion_top_k)
    submission_predictions = {lang: [] for lang in languages}

    for path in result_paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                _record_query_result(
                    item=item,
                    metrics=metrics,
                    submission_predictions=(
                        submission_predictions if export_submission_tsv else None
                    ),
                )

    if export_submission_tsv:
        for lang_predictions in submission_predictions.values():
            lang_predictions.sort(key=lambda row: row["index"])
        write_submission_tsv_files(submission_predictions, submission_output_dir)

    summary = metrics.summary()
    if metrics_output_file:
        Path(metrics_output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_output_file, "w", encoding="utf-8") as handle:
            json.dump(summary["languages"], handle, indent=2)

    return {
        "summary": summary,
        "submission_predictions": submission_predictions
        if export_submission_tsv
        else {},
    }


def _build_base_payload(args: argparse.Namespace) -> dict[str, Any]:
    split = normalize_split(args.split)
    languages = _parse_csv(args.languages)
    if not languages:
        raise ValueError("--languages must include at least one language code.")
    return {
        "split": split,
        "languages": languages,
        "cache_dir": args.cache_dir,
        "profile": args.profile,
        "fusion_method": args.fusion_method,
        "fusion_top_k": args.fusion_top_k,
        "force_retrain_fusion": args.force_retrain_fusion,
        "hf_fusion_repo_id": args.hf_fusion_repo_id,
        "dense_model": args.dense_model,
        "disable_sparse": args.disable_sparse,
        "reranker_model": args.reranker_model,
        "disable_reranker": args.disable_reranker,
        "sparse_vanilla": args.sparse_vanilla,
        "custom_papers": args.custom_papers,
        "force_recompute_sparse_cache": args.force_recompute_sparse_cache,
        "force_recompute_dense_documents": args.force_recompute_dense_documents,
        "force_recompute_dense_queries": args.force_recompute_dense_queries,
        "max_queries": args.max_queries,
    }


def _write_worker_payloads(
    *,
    base_payload: dict[str, Any],
    work_dir: str,
    num_workers: int,
    chunk_size: int | None,
) -> list[dict[str, str]]:
    payload_dir = Path(work_dir, "payloads")
    result_dir = Path(work_dir, "chunks")
    payload_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, str]] = []
    split = base_payload["split"]
    for lang in base_payload["languages"]:
        tweets = load_query_split(lang, split)
        if base_payload.get("max_queries"):
            tweets = tweets[: int(base_payload["max_queries"])]
        if chunk_size:
            groups = [
                list(range(start, end))
                for start, end in _chunk_indices(len(tweets), chunk_size)
            ]
        else:
            groups = _split_indices(len(tweets), num_workers)

        for chunk_idx, query_indices in enumerate(groups):
            result_path = result_dir / f"{lang}_{chunk_idx:03d}.jsonl"
            payload_path = payload_dir / f"{lang}_{chunk_idx:03d}.json"
            payload = {
                **base_payload,
                "lang": lang,
                "query_indices": query_indices,
                "output_path": str(result_path),
            }
            with payload_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            tasks.append(
                {"payload_path": str(payload_path), "result_path": str(result_path)}
            )

    return tasks


def _run_worker_wave(
    *,
    wave: list[dict[str, str]],
    gpu_ids: list[str],
) -> None:
    processes = []
    for worker_idx, task in enumerate(wave):
        env = os.environ.copy()
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_idx % len(gpu_ids)]
        command = [
            sys.executable,
            "-m",
            "clef_pipeline.evaluate_language",
            "--worker-payload",
            task["payload_path"],
        ]
        print(
            "Launching worker "
            f"{task['payload_path']} on CUDA_VISIBLE_DEVICES="
            f"{env.get('CUDA_VISIBLE_DEVICES', '<unchanged>')}"
        )
        processes.append(subprocess.Popen(command, env=env))

    failures = []
    for task, process in zip(wave, processes, strict=True):
        return_code = process.wait()
        if return_code != 0:
            failures.append((task["payload_path"], return_code))
    if failures:
        details = ", ".join(f"{path} exited {code}" for path, code in failures)
        raise RuntimeError(f"Evaluation worker failure: {details}")


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare caches, launch parallel workers, and aggregate metrics."""
    base_payload = _build_base_payload(args)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare_cache:
        prepare_evaluation_cache(base_payload)

    tasks = _write_worker_payloads(
        base_payload=base_payload,
        work_dir=args.work_dir,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )
    gpu_ids = _parse_csv(args.gpu_ids)
    if not gpu_ids and args.num_workers > 1:
        raise ValueError("--gpu-ids must be non-empty when --num-workers > 1.")

    for wave_start in range(0, len(tasks), args.num_workers):
        wave = tasks[wave_start : wave_start + args.num_workers]
        _run_worker_wave(wave=wave, gpu_ids=gpu_ids)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    submission_output_dir = args.submission_output_dir or str(
        Path(args.work_dir, "submissions", f"{base_payload['split']}-{run_id}")
    )
    output = _aggregate_results(
        result_paths=[task["result_path"] for task in tasks],
        languages=base_payload["languages"],
        fusion_top_k=base_payload["fusion_top_k"],
        export_submission_tsv=args.export_submission_tsv,
        submission_output_dir=submission_output_dir,
        metrics_output_file=args.metrics_output_file,
    )

    fusion_label = (
        "RF Fusion" if base_payload["fusion_method"] == "random_forest" else "RRF"
    )
    for lang in base_payload["languages"]:
        _print_language_summary(
            lang,
            output["summary"]["languages"][lang],
            base_payload["fusion_top_k"],
            fusion_label,
        )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-payload")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--languages", default=DEFAULT_LANGUAGES)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-ids", default=DEFAULT_GPU_IDS)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--max-queries", type=int)

    parser.add_argument("--profile", default="custom")
    parser.add_argument("--dense-model", default="harrier-27b")
    parser.add_argument("--disable-sparse", action="store_true")
    parser.add_argument("--fusion-method", default="rrf")
    parser.add_argument("--fusion-top-k", type=int, default=60)
    parser.add_argument("--force-retrain-fusion", action="store_true")
    parser.add_argument(
        "--hf-fusion-repo-id",
        default="boyes-boys-clef-2026/random-forest-fuser",
    )
    parser.add_argument("--reranker-model", default="nemotron")
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--sparse-vanilla", action="store_true")
    parser.add_argument("--custom-papers")

    parser.add_argument("--force-recompute-sparse-cache", action="store_true")
    parser.add_argument("--force-recompute-dense-documents", action="store_true")
    parser.add_argument("--force-recompute-dense-queries", action="store_true")
    parser.add_argument("--prepare-cache", dest="prepare_cache", action="store_true")
    parser.add_argument(
        "--no-prepare-cache", dest="prepare_cache", action="store_false"
    )
    parser.set_defaults(prepare_cache=True)

    parser.add_argument("--metrics-output-file")
    parser.add_argument("--export-submission-tsv", action="store_true")
    parser.add_argument("--submission-output-dir")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker_payload:
        evaluate_worker_payload(args.worker_payload)
        return
    run_controller(args)


if __name__ == "__main__":
    main()
