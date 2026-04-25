"""Train Random Forest fuser locally using cached embeddings.

Uses the real pipeline classes (HarrierRetriever, SparseRetriever,
RandomForestFuser) — the Harrier model is never loaded, only its
cached embeddings are read from disk onto CPU.

Prerequisites:
  1. Document embeddings downloaded from Modal volume:
     uv run modal volume get checkthat-embedding-cache \
       microsoft--harrier-oss-v1-27b/ cache/dense_train_local/microsoft--harrier-oss-v1-27b/

  2. Dense train queries cached via modal_dense_train_cache.py:
     cache/dense_train_local/microsoft--harrier-oss-v1-27b/queries_train_{lang}-{fp}.pt

  3. Sparse train queries cached via local_sparse_train_cache.py:
     cache/sparse_train_local/sparse/bm25plus-.../sparse_queries_train_{lang}-...npz

Usage:
    uv run python scripts/local_rf_train.py
    uv run python scripts/local_rf_train.py --skip-hf --skip-modal
"""

import hashlib
import os
import time

import torch
from datasets import load_dataset
from dotenv import load_dotenv

from clef_pipeline.fusions import RandomForestFuser
from clef_pipeline.interfaces import BaseReranker
from clef_pipeline.retrievers import HarrierRetriever, SparseRetriever
from clef_pipeline.utils import CHECKTHAT_DATASET

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — must match pipeline settings exactly
# ---------------------------------------------------------------------------
HARRIER_MODEL_NAME = "microsoft/harrier-oss-v1-27b"
HARRIER_CACHE_KEY = HARRIER_MODEL_NAME.replace("/", "--")
SPARSE_K1 = 2.5
SPARSE_B = 0.85
SPARSE_USE_BIGRAMS = True
SPARSE_USE_TRANSLATION = True
SPARSE_CACHE_TOP_K = 2000
DEFAULT_DENSE_DOC_MAX_CHARS = 2048
LANGUAGES = ["de", "fr", "en"]

DENSE_CACHE_DIR = os.path.join("cache", "dense_train_local")
SPARSE_CACHE_DIR = os.path.join("cache", "sparse_train_local")
RF_OUTPUT_DIR = os.path.join("cache", "rf_model_local")

MODAL_VOLUME_NAME = "checkthat-embedding-cache"
HF_REPO_ID = "boyes-boys-clef-2026/random-forest-fuser"


def _texts_fingerprint(texts: list[str]) -> str:
    """Same as HarrierRetriever._texts_fingerprint."""
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8", errors="ignore")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()[:16]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train RF fuser locally.")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace upload.")
    parser.add_argument("--skip-modal", action="store_true", help="Skip Modal volume upload.")
    args = parser.parse_args()

    overall_start = time.time()

    # -----------------------------------------------------------------------
    # 1. Load collection
    # -----------------------------------------------------------------------
    print("Loading collection from HuggingFace...")
    collection = load_dataset(CHECKTHAT_DATASET, "collection", split="collection")
    collection_docs = collection.to_list()
    article_pubkeys = [doc["pubkey"] for doc in collection_docs]
    article_texts = [
        BaseReranker.document_to_text(doc)[:DEFAULT_DENSE_DOC_MAX_CHARS]
        for doc in collection_docs
    ]
    doc_fp = _texts_fingerprint(article_texts)
    print(f"  {len(collection_docs)} documents, fingerprint: {doc_fp}")

    # -----------------------------------------------------------------------
    # 2. Set up HarrierRetriever (no model loaded, only cached embeddings)
    # -----------------------------------------------------------------------
    print("\nSetting up Harrier retriever (CPU, no model)...")
    dense_retriever = HarrierRetriever()

    # Load document embeddings onto CPU
    doc_emb_filename = f"documents-{doc_fp}.pt"
    doc_emb_path = os.path.join(DENSE_CACHE_DIR, HARRIER_CACHE_KEY, doc_emb_filename)
    if not os.path.exists(doc_emb_path):
        raise FileNotFoundError(
            f"Document embeddings not found at {doc_emb_path}\n\n"
            f"Download with:\n"
            f"  uv run modal volume get {MODAL_VOLUME_NAME} "
            f"{HARRIER_CACHE_KEY}/{doc_emb_filename} {doc_emb_path}"
        )
    dense_retriever.embeddings = torch.load(
        doc_emb_path, map_location="cuda", weights_only=True
    )
    print(f"  Loaded doc embeddings: {dense_retriever.embeddings.shape}")

    # -----------------------------------------------------------------------
    # 3. Set up SparseRetriever (index collection to get corpus fingerprint)
    # -----------------------------------------------------------------------
    print("\nSetting up sparse retriever (indexing collection)...")
    sparse_retriever = SparseRetriever(
        k1=SPARSE_K1,
        b=SPARSE_B,
        use_bigrams=SPARSE_USE_BIGRAMS,
        use_translation=SPARSE_USE_TRANSLATION,
    )
    t0 = time.time()
    sparse_retriever.index(collection_docs)
    print(f"  Indexed in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # 4. Load train data + cached embeddings per language
    # -----------------------------------------------------------------------
    train_tweets_by_lang: dict[str, list[dict]] = {}

    for lang in LANGUAGES:
        print(f"\n  Loading {lang.upper()} train data...")
        train_tweets = list(load_dataset(CHECKTHAT_DATASET, lang)["train"])
        train_tweets_by_lang[lang] = train_tweets
        query_texts = [row["text"] for row in train_tweets]
        print(f"    {len(train_tweets)} tweets")

        # Dense: load cached query embeddings onto CPU
        query_fp = _texts_fingerprint(query_texts)
        dense_q_path = os.path.join(
            DENSE_CACHE_DIR, HARRIER_CACHE_KEY,
            f"queries_train_{lang}-{query_fp}.pt",
        )
        if not os.path.exists(dense_q_path):
            raise FileNotFoundError(
                f"Dense train queries not found: {dense_q_path}\n"
                f"Run: uv run modal run scripts/modal_dense_train_cache.py"
            )
        embs = torch.load(dense_q_path, map_location="cuda", weights_only=True)
        dense_retriever._query_embeddings_by_name[f"queries_train_{lang}"] = embs
        print(f"    Dense: loaded {embs.shape[0]} query embeddings")

        # Sparse: use real index_queries() — will hit the .npz cache file
        sparse_retriever.index_queries(
            query_texts,
            lang=lang,
            cache_dir=SPARSE_CACHE_DIR,
            cache_name=f"sparse_queries_train_{lang}",
            top_k=SPARSE_CACHE_TOP_K,
        )

    # -----------------------------------------------------------------------
    # 5. Train Random Forest Fuser
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Training Random Forest Fuser")
    print(f"{'=' * 60}")

    hf_token = os.environ.get("HUGGING_FACE")
    fuser = RandomForestFuser(
        hf_repo_id=None if args.skip_hf else HF_REPO_ID,
        hf_token=hf_token,
    )
    sparse_config = {"k1": SPARSE_K1, "b": SPARSE_B, "stemmer": "lancaster"}

    fuser.train(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        train_tweets_by_lang=train_tweets_by_lang,
        article_pubkeys=article_pubkeys,
        dense_model_name=HARRIER_MODEL_NAME,
        sparse_config=sparse_config,
        train_split="train",
    )

    # -----------------------------------------------------------------------
    # 6. Save locally + upload to HuggingFace (handled by fuser.save())
    # -----------------------------------------------------------------------
    os.makedirs(RF_OUTPUT_DIR, exist_ok=True)
    saved_path = fuser.save(cache_dir=RF_OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # 7. Upload to Modal volume
    # -----------------------------------------------------------------------
    if not args.skip_modal:
        import modal

        print(f"\nUploading to Modal volume '{MODAL_VOLUME_NAME}'...")
        volume = modal.Volume.from_name(MODAL_VOLUME_NAME)
        remote_path = f"/rf_fusion_models/{os.path.basename(saved_path)}"
        with open(saved_path, "rb") as f:
            volume.write_file(remote_path, f)
        volume.commit()
        size_mb = os.path.getsize(saved_path) / (1024 * 1024)
        print(f"  Uploaded to {remote_path} ({size_mb:.1f} MB)")

    total = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"  Complete — {total:.0f}s total")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
