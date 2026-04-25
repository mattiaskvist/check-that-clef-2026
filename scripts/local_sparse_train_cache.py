"""Build sparse retriever cache for RF training data locally (no GPU needed).

Indexes the train split through the BM25+ sparse retriever and saves the
.npz cache files locally, then uploads them to the Modal volume so the
cloud pipeline gets cache hits when training the Random Forest fuser.

Usage:
    uv run python scripts/local_sparse_train_cache.py
    uv run python scripts/local_sparse_train_cache.py --upload
"""

import argparse
import glob
import os
import time

from datasets import load_dataset
from tqdm import tqdm

from clef_pipeline.retrievers import SparseRetriever
from clef_pipeline.utils import CHECKTHAT_DATASET

# Must match pipeline settings in pipeline_config.py / main.py
SPARSE_K1 = 2.5
SPARSE_B = 0.85
SPARSE_USE_BIGRAMS = True
SPARSE_USE_TRANSLATION = True
SPARSE_CACHE_TOP_K = 2000
LANGUAGES = ["de", "fr", "en"]
LOCAL_CACHE_DIR = os.path.join("cache", "sparse_train_local")
MODAL_VOLUME_NAME = "checkthat-embedding-cache"


def main():
    parser = argparse.ArgumentParser(
        description="Build sparse train query cache locally and upload to Modal."
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload cache files to the Modal volume after building.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if local cache files already exist.",
    )
    args = parser.parse_args()

    overall_start = time.time()

    # Initialize sparse retriever (CPU only)
    print("Initializing sparse retriever...")
    retriever = SparseRetriever(
        k1=SPARSE_K1,
        b=SPARSE_B,
        use_bigrams=SPARSE_USE_BIGRAMS,
        use_translation=SPARSE_USE_TRANSLATION,
    )

    # Load and index collection
    print("Loading collection from HuggingFace...")
    collection = load_dataset(CHECKTHAT_DATASET, "collection", split="collection")
    collection_documents = collection.to_list()

    print(f"Indexing {len(collection_documents)} documents...")
    t0 = time.time()
    retriever.index(collection_documents)
    print(f"Collection indexed in {time.time() - t0:.1f}s")

    # Process each language (train split only)
    for lang in LANGUAGES:
        print(f"\n{'=' * 60}")
        print(f"  {lang.upper()} — loading train split")
        print(f"{'=' * 60}")

        train_tweets = list(load_dataset(CHECKTHAT_DATASET, lang)["train"])
        print(f"  {len(train_tweets)} train tweets")

        query_texts = [row["text"] for row in train_tweets]
        cache_name = f"sparse_queries_train_{lang}"

        # Check if cache already exists
        cache_path = retriever._cache_path(
            LOCAL_CACHE_DIR, cache_name, query_texts, lang, SPARSE_CACHE_TOP_K
        )
        if cache_path and os.path.exists(cache_path) and not args.force:
            print(f"  [skip] Cache already exists: {cache_path}")
            continue

        print(f"  Scoring {len(query_texts)} queries (this may take a while)...")
        t0 = time.time()

        # Score queries with progress bar
        corpus_size = len(collection_documents)
        effective_top_k = min(SPARSE_CACHE_TOP_K, corpus_size)

        import numpy as np

        rankings = np.empty((len(query_texts), effective_top_k), dtype=np.int32)
        scores = np.empty((len(query_texts), effective_top_k), dtype=np.float32)

        for i, query_text in enumerate(
            tqdm(query_texts, desc=f"  Sparse {lang.upper()}", unit="query")
        ):
            ranked_indices, ranked_scores = retriever._score_query(
                query_text, lang=lang, top_k=effective_top_k
            )
            rankings[i] = ranked_indices
            scores[i] = ranked_scores

        # Store in retriever's internal cache (for consistency)
        retriever._query_rankings_by_name[cache_name] = rankings
        retriever._query_scores_by_name[cache_name] = scores

        # Save to disk
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, rankings=rankings, scores=scores)
            size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"  Saved to {cache_path} ({size_mb:.1f} MB)")

        elapsed = time.time() - t0
        qps = len(query_texts) / elapsed if elapsed > 0 else 0
        print(f"  Done in {elapsed:.1f}s ({qps:.1f} queries/sec)")

    # Summary
    sparse_dir = os.path.join(LOCAL_CACHE_DIR, "sparse")
    cache_files = glob.glob(os.path.join(sparse_dir, "**", "*.npz"), recursive=True)

    total_elapsed = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"  Complete — {total_elapsed:.0f}s total")
    print(f"{'=' * 60}")
    for f in cache_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {os.path.basename(f)} ({size_mb:.1f} MB)")

    if args.upload:
        _upload_to_modal(sparse_dir)
    else:
        print(f"\nTo upload to Modal volume, re-run with --upload or run:")
        print(f"  uv run modal volume put {MODAL_VOLUME_NAME} {sparse_dir}/ /sparse/")


def _upload_to_modal(sparse_dir: str):
    """Upload cache files to the Modal volume using the Modal Python SDK."""
    import modal

    print(f"\nUploading to Modal volume '{MODAL_VOLUME_NAME}'...")
    volume = modal.Volume.from_name(MODAL_VOLUME_NAME)

    cache_files = glob.glob(os.path.join(sparse_dir, "**", "*.npz"), recursive=True)
    for local_path in cache_files:
        rel_path = os.path.relpath(local_path, sparse_dir)
        remote_path = f"/sparse/{rel_path}".replace("\\", "/")

        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  {rel_path} -> {remote_path} ({size_mb:.1f} MB)")
        with open(local_path, "rb") as f:
            volume.write_file(remote_path, f)

    volume.commit()
    print("Upload complete.")


if __name__ == "__main__":
    main()
