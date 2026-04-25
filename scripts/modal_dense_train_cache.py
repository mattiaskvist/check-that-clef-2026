"""Cache dense Harrier-27b embeddings for RF training data on Modal (GPU).

Encodes the train-split queries through the HarrierRetriever on a Modal GPU,
saves the .pt cache files to the shared embedding volume, and downloads them
locally for offline RF training.

Document embeddings are NOT computed — they must already exist on the volume
from a previous evaluation run.

Usage:
    uv run modal run scripts/modal_dense_train_cache.py
    uv run modal run scripts/modal_dense_train_cache.py --force-recompute
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "sentence-transformers <= 5.3.0",
        "peft",
        "datasets",
        "tqdm",
        "transformers <= 5.5.1",
        "accelerate",
        "nltk",
        "deep-translator",
        "huggingface-hub <=	1.9.2",
    )
    .add_local_python_source("clef_pipeline")
)

app = modal.App("checkthat-dense-train-cache")
EMBEDDING_CACHE_VOLUME_NAME = "checkthat-embedding-cache"
embedding_cache = modal.Volume.from_name(
    EMBEDDING_CACHE_VOLUME_NAME, create_if_missing=True
)
CACHE_MOUNT = "/cache/embeddings"
LANGUAGES = ["de", "fr", "en"]
LOCAL_CACHE_DIR = "cache/dense_train_local"


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 3,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={CACHE_MOUNT: embedding_cache},
)
def cache_dense_train_embeddings(force_recompute: bool = False):
    """Encode and cache Harrier-27b train query embeddings."""
    import time

    from datasets import load_dataset

    from clef_pipeline.retrievers import HarrierRetriever
    from clef_pipeline.utils import CHECKTHAT_DATASET

    overall_start = time.time()

    retriever = HarrierRetriever()
    print(f"Using model: {retriever.model_name}")

    saved_files: list[dict] = []

    for lang in LANGUAGES:
        print(f"\n{'=' * 60}")
        print(f"  {lang.upper()} — encoding TRAIN queries")
        print(f"{'=' * 60}")

        train_tweets = list(load_dataset(CHECKTHAT_DATASET, lang)["train"])
        query_texts = [row["text"] for row in train_tweets]
        print(f"  {len(query_texts)} train queries")

        cache_name = f"queries_train_{lang}"

        t0 = time.time()
        retriever.index_queries(
            query_texts,
            cache_dir=CACHE_MOUNT,
            cache_name=cache_name,
            force_recompute=force_recompute,
        )
        elapsed = time.time() - t0
        qps = len(query_texts) / elapsed if elapsed > 0 else 0
        print(f"  Done in {elapsed:.1f}s ({qps:.1f} queries/sec)")

        # Track the path that was written
        cache_path = retriever._cache_path(CACHE_MOUNT, f"{cache_name}.pt", query_texts)
        # Remote path relative to volume root
        remote_path = cache_path.replace(CACHE_MOUNT, "").lstrip("/")
        saved_files.append({"lang": lang, "remote_path": remote_path})

        embedding_cache.commit()

    total = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"  Complete — {total:.0f}s total")
    print(f"{'=' * 60}")
    for f in saved_files:
        print(f"  {f['lang'].upper()}: {f['remote_path']}")

    return saved_files


@app.local_entrypoint()
def main(force_recompute: bool = False):
    """Run GPU encoding on Modal, then download results locally."""
    import os

    saved_files = cache_dense_train_embeddings.remote(force_recompute=force_recompute)

    # Download from Modal volume to local cache
    print(f"\nDownloading to {LOCAL_CACHE_DIR}/...")
    volume = modal.Volume.from_name(EMBEDDING_CACHE_VOLUME_NAME)

    for f in saved_files:
        remote_path = f["remote_path"]
        local_path = os.path.join(LOCAL_CACHE_DIR, remote_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"  {f['lang'].upper()}: {remote_path} -> {local_path}")
        data = b"".join(volume.read_file(remote_path))
        with open(local_path, "wb") as fh:
            fh.write(data)
        size_mb = len(data) / (1024 * 1024)
        print(f"        {size_mb:.1f} MB")

    print(f"\nDone. Local cache: {LOCAL_CACHE_DIR}/")
