"""Local runner for optimized sparse retrieval (no Modal required).

This script is meant for quick iteration on SparseRetriever logic and its impact
on metrics (e.g., German MRR@5) without standing up Modal GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="dev",
        choices=["train", "dev", "test"],
        help="Dataset split to run.",
    )
    parser.add_argument(
        "--lang",
        default="de",
        choices=["de", "en", "fr"],
        help="Language to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Max ranked docs to score per query.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional path to write JSON summary.",
    )
    parser.add_argument(
        "--sparse-vanilla",
        action="store_true",
        help="Run the vanilla BM25 configuration (no bigrams/translation, weaker params).",
    )
    parser.add_argument(
        "--compare-translation-overlap",
        action="store_true",
        help=(
            "Run sparse twice (translation on/off) and report top-k overlap stats "
            "(intersection size, Jaccard, unique counts)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Thread workers for query evaluation. Defaults to env NUM_WORKER(S) or 1.",
    )
    return parser.parse_args()


def main() -> int:
    from datasets import load_dataset
    from tqdm import tqdm

    from clef_pipeline.metrics import EvaluationMetrics
    from clef_pipeline.retrievers import SparseRetriever
    from clef_pipeline.utils import CHECKTHAT_DATASET, load_query_split

    args = _parse_args()
    env_workers = os.environ.get("NUM_WORKERS") or os.environ.get("NUM_WORKER")
    num_workers = args.num_workers or (int(env_workers) if env_workers else 1)
    num_workers = max(1, int(num_workers))

    sparse_k1 = 1.5 if args.sparse_vanilla else 2.5
    sparse_b = 0.75 if args.sparse_vanilla else 0.85
    sparse_use_bigrams = not args.sparse_vanilla
    sparse_use_translation = not args.sparse_vanilla

    print("Loading collection...")
    collection = load_dataset(CHECKTHAT_DATASET, "collection", split="collection").to_list()
    pubkeys = [str(doc.get("pubkey")) for doc in collection]

    print("Indexing collection for sparse retrieval...")
    # Reuse the same BM25 index; translation only affects query preprocessing.
    retriever = SparseRetriever(
        k1=sparse_k1,
        b=sparse_b,
        use_bigrams=sparse_use_bigrams,
        use_translation=sparse_use_translation,
    )
    retriever.index(collection)

    print("Indexing no-translation sparse retriever...")
    retriever_no_trans = SparseRetriever(
        k1=sparse_k1,
        b=sparse_b,
        use_bigrams=sparse_use_bigrams,
        use_translation=False,
    )
    retriever_no_trans.index(collection)

    print(f"Loading queries ({args.lang}/{args.split})...")
    queries = load_query_split(args.lang, args.split)

    if args.compare_translation_overlap:
        from statistics import mean, median

        no_trans = SparseRetriever(
            k1=sparse_k1,
            b=sparse_b,
            use_bigrams=sparse_use_bigrams,
            use_translation=False,
        )

        print("Indexing no-translation retriever...")
        no_trans.index(collection)

        with_trans = SparseRetriever(
            k1=sparse_k1,
            b=sparse_b,
            use_bigrams=sparse_use_bigrams,
            use_translation=True,
        )

        print("Sharing BM25 index with translation retriever...")

        # Share immutable indexed structures
        with_trans.bm25_model = no_trans.bm25_model
        with_trans._docs_tokens = no_trans._docs_tokens
        with_trans._term_graph = no_trans._term_graph
        with_trans._indexed_corpus_fingerprint = (
            no_trans._indexed_corpus_fingerprint
        )

        per_query: list[dict] = []

        def _rrf_fuse(a: list[int], b: list[int], k: int = 60) -> list[int]:
            scores = {}

            for rank, doc_id in enumerate(a):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

            for rank, doc_id in enumerate(b):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

            return [
                doc_id
                for doc_id, _ in sorted(
                    scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]

        def _overlap_one(query_idx: int, row: dict) -> dict:
            query_text = row["text"]

            no_trans_results = no_trans.search(
                query_text,
                lang=args.lang,
            )[: args.top_k]

            trans_results = with_trans.search(
                query_text,
                lang=args.lang,
            )[: args.top_k]

            fused_results = _rrf_fuse(
                no_trans_results,
                trans_results,
            )[: args.top_k]

            set_a = set(int(x) for x in no_trans_results)
            set_b = set(int(x) for x in trans_results)
            set_fused = set(int(x) for x in fused_results)

            inter = set_a & set_b
            union = set_a | set_b

            return {
                "query_idx": query_idx,

                "intersection": len(inter),
                "union": len(union),

                "jaccard": (
                    len(inter) / len(union)
                    if union else 0.0
                ),

                "unique_no_translation": len(set_a - set_b),
                "unique_translation": len(set_b - set_a),

                "fused_unique_docs": len(set_fused),
            }

        if num_workers == 1:
            for idx, row in enumerate(tqdm(queries, desc="Comparing")):
                per_query.append(_overlap_one(idx, row))
        else:
            print(f"Comparing with {num_workers} worker threads...")

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(_overlap_one, idx, row): idx
                    for idx, row in enumerate(queries)
                }

                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Comparing",
                ):
                    per_query.append(fut.result())

            per_query.sort(key=lambda x: x["query_idx"])

        intersections = [x["intersection"] for x in per_query]
        jaccards = [x["jaccard"] for x in per_query]
        uniq_a = [x["unique_no_translation"] for x in per_query]
        uniq_b = [x["unique_translation"] for x in per_query]
        unions = [x["union"] for x in per_query]

        report = {
            "lang": args.lang,
            "split": args.split,
            "top_k": args.top_k,
            "n_queries": len(per_query),

            "intersection": {
                "mean": mean(intersections) if intersections else 0.0,
                "median": median(intersections) if intersections else 0.0,
                "min": min(intersections) if intersections else 0,
                "max": max(intersections) if intersections else 0,
            },

            "union": {
                "mean": mean(unions) if unions else 0.0,
                "median": median(unions) if unions else 0.0,
                "min": min(unions) if unions else 0,
                "max": max(unions) if unions else 0,
            },

            "jaccard": {
                "mean": mean(jaccards) if jaccards else 0.0,
                "median": median(jaccards) if jaccards else 0.0,
                "min": min(jaccards) if jaccards else 0.0,
                "max": max(jaccards) if jaccards else 0.0,
            },

            "unique_no_translation": {
                "mean": mean(uniq_a) if uniq_a else 0.0,
                "median": median(uniq_a) if uniq_a else 0.0,
            },

            "unique_translation": {
                "mean": mean(uniq_b) if uniq_b else 0.0,
                "median": median(uniq_b) if uniq_b else 0.0,
            },
        }

        print(json.dumps(report, indent=2, sort_keys=True))

        if args.metrics_output:
            args.metrics_output.parent.mkdir(parents=True, exist_ok=True)

            args.metrics_output.write_text(
                json.dumps(report, indent=2, sort_keys=True)
            )

            print(f"Wrote overlap JSON to: {args.metrics_output}")

        return 0

    metrics = EvaluationMetrics(fusion_top_k=args.top_k)

    def _build_custom_fusion(
        sparse_combined: list[int],
        sparse_untranslated: list[int],
        top_k: int = 200,
        seed_k: int = 50,
        untranslated_k: int = 50,
    ) -> list[int]:
        """
        Fusion strategy:
        1. top-50 from translated sparse
        2. top-50 from untranslated sparse (unique only)
        3. remaining from translated sparse (unique only)
        cap at 200 total
        """
        seen = set()
        fused = []

        # 1) seed from translated sparse
        for doc in sparse_combined[:seed_k]:
            if doc not in seen:
                seen.add(doc)
                fused.append(doc)

        # 2) add untranslated top-50
        for doc in sparse_untranslated[:untranslated_k]:
            if doc not in seen:
                seen.add(doc)
                fused.append(doc)

        # 3) fill remainder from translated sparse (full list continuation)
        for doc in sparse_combined:
            if doc not in seen:
                seen.add(doc)
                fused.append(doc)
            if len(fused) >= top_k:
                break

        return fused[:top_k]

    def _eval_one(query_idx: int, row: dict) -> tuple[int, str, dict[str, list[str]]]:
        query_text = row["text"]

        # -----------------------------
        # INDEPENDENT SPARSE RUNS
        # -----------------------------
        sparse_combined = retriever.search(
            query_text,
            lang=args.lang,
        )[: args.top_k]

        sparse_untranslated = retriever_no_trans.search(
            query_text,
            lang=args.lang,
        )[: args.top_k]

        # -----------------------------
        # STRICT FUSION (NO DENSE)
        # -----------------------------
        final_ranked = _build_custom_fusion(
            sparse_combined=sparse_combined,
            sparse_untranslated=sparse_untranslated,
            top_k=200,
        )

        # -----------------------------
        # MAP TO PUBKEYS
        # -----------------------------
        preds = []
        for doc_id in final_ranked:
            if 0 <= doc_id < len(pubkeys):
                preds.append(pubkeys[doc_id])

        stages = {
            "sparse_combined": sparse_combined,
            "sparse_untranslated": sparse_untranslated,
            "final": preds,
        }

        true_pubkey = str(row.get("pubkey") or "")
        return query_idx, true_pubkey, stages

    if num_workers == 1:
        for idx, row in enumerate(tqdm(queries, desc="Scoring queries")):
            _, true_pubkey, stages = _eval_one(idx, row)
            metrics.add_query(lang=args.lang, true_pubkey=true_pubkey, stages=stages)
    else:
        print(f"Evaluating with {num_workers} worker threads...")
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(_eval_one, idx, row): idx for idx, row in enumerate(queries)
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
                _, true_pubkey, stages = fut.result()
                metrics.add_query(lang=args.lang, true_pubkey=true_pubkey, stages=stages)

    summary = metrics.summary()
    lang_summary = summary["languages"].get(args.lang, {})
    print(json.dumps(lang_summary, indent=2, sort_keys=True))

    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote metrics JSON to: {args.metrics_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
