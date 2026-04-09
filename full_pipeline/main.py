import modal

from .rerankers import NemotronReranker, Gemma2BReranker
from .retrievers import HarrierRetriever, SparseRetriever, BGEM3Retriever
from .utils import CHECKTHAT_DATASET, FusionProcessor, MRR_at_5, article_to_text
from .fusions import ScoreFusionProcessor

# ==========================================
# 1. MODAL ENVIRONMENT SETUP
# ==========================================
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
        "xgboost",
        "scikit-learn",
    )
)

app = modal.App("checkthat-evaluation-pipeline")
embedding_cache = modal.Volume.from_name(
    "checkthat-embedding-cache", create_if_missing=True
)

CACHE_MOUNT = "/cache/embeddings"


# ==========================================
# 2. MAIN CLOUD FUNCTION
# ==========================================
@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60 * 3,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={CACHE_MOUNT: embedding_cache},
)
def evaluate_pipeline():
    import numpy as np
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    # --- INITIALIZE COMPONENTS ---
    dense_retriever = BGEM3Retriever()
    sparse_retriever = SparseRetriever()
    reranker = Gemma2BReranker()
    fusion = FusionProcessor()
    score_fusion = ScoreFusionProcessor()

    # --- LOAD & INDEX COLLECTION ---
    collection_dataset = load_dataset(
        CHECKTHAT_DATASET, "collection", split="collection"
    )
    article_texts = [article_to_text(doc) for doc in collection_dataset]
    article_pubkeys = [doc["pubkey"] for doc in collection_dataset]

    dense_retriever.index(article_texts, cache_dir=CACHE_MOUNT)
    embedding_cache.commit()
    sparse_retriever.index(collection_dataset)

    # --- 3. PRE-ENCODE ALL QUERIES, THEN FREE EMBEDDING MODEL ---
    languages = ["de", "fr", "en"]
    lang_tweets = {}
    for lang in languages:
        tweets = list(load_dataset(CHECKTHAT_DATASET, lang)["dev"])
        lang_tweets[lang] = tweets
        query_texts = [row["text"] for row in tweets]
        dense_retriever.index_queries(
            query_texts, cache_dir=CACHE_MOUNT, cache_name=f"queries_{lang}"
        )
        embedding_cache.commit()

    dense_retriever.unload_model()

    # --- 4. EVALUATION LOOP (collect scores for all systems) ---
    global_results = {}
    global_totals = {
        "queries": 0,
        "dense_sum": 0.0,
        "sparse_sum": 0.0,
        "rrf_sum": 0.0,
        "final_sum": 0.0,
    }

    # Storage for XGBoost data collection
    all_xgb_data = []  # list of dicts per query

    for lang in languages:
        print("\n==========================================")
        print(f"  STARTING EVALUATION FOR LANGUAGE: {lang.upper()}")
        print("==========================================")

        tweets = lang_tweets[lang]

        dense_mrr, sparse_mrr, rrf_mrr, final_mrr = [], [], [], []

        for i, row in enumerate(
            tqdm(tweets, desc=f"Evaluating {lang.upper()} Queries")
        ):
            query_text = row["text"]
            true_pubkey = row["pubkey"]

            # Step A: Independent Retrieval (now returns scores too)
            dense_ranks, dense_scores = dense_retriever.search(
                i, cache_name=f"queries_{lang}"
            )
            sparse_ranks, sparse_scores = sparse_retriever.search(query_text)

            dense_mrr.append(
                MRR_at_5(
                    [article_pubkeys[doc_id] for doc_id in dense_ranks], true_pubkey
                )
            )
            sparse_mrr.append(
                MRR_at_5(
                    [article_pubkeys[doc_id] for doc_id in sparse_ranks], true_pubkey
                )
            )

            # Step B: Fusion (now returns scores too)
            fused_candidates, rrf_scores = fusion.reciprocal_rank_fusion(
                [dense_ranks, sparse_ranks], top_k=10
            )
            rrf_mrr.append(
                MRR_at_5(
                    [article_pubkeys[doc_id] for doc_id in fused_candidates],
                    true_pubkey,
                )
            )

            # Step C: Reranking
            final_results = reranker.rerank(query_text, fused_candidates, article_texts)
            final_mrr.append(
                MRR_at_5(
                    [article_pubkeys[doc_id] for doc_id, score in final_results],
                    true_pubkey,
                )
            )

            # Step D: Collect features for XGBoost
            features = ScoreFusionProcessor.build_query_features(
                candidates=fused_candidates,
                dense_scores=dense_scores,
                dense_ranked=dense_ranks,
                sparse_scores=sparse_scores,
                sparse_ranked=sparse_ranks,
                rrf_scores=rrf_scores,
                rrf_ranked=fused_candidates,
                reranker_results=final_results,
            )

            # Labels: 1 if candidate is the correct paper, else 0
            labels = [
                1 if article_pubkeys[doc_id] == true_pubkey else 0
                for doc_id in fused_candidates
            ]

            all_xgb_data.append(
                {
                    "lang": lang,
                    "query_idx": i,
                    "true_pubkey": true_pubkey,
                    "candidates": fused_candidates,
                    "candidate_pubkeys": [
                        article_pubkeys[doc_id] for doc_id in fused_candidates
                    ],
                    "features": features,
                    "labels": labels,
                    # Store per-system MRRs for this query for test-split comparison
                    "dense_mrr": dense_mrr[-1],
                    "sparse_mrr": sparse_mrr[-1],
                    "rrf_mrr": rrf_mrr[-1],
                    "rerank_mrr": final_mrr[-1],
                }
            )

        # Track and print metrics (UNCHANGED from original)
        avg_dense = sum(dense_mrr) / len(dense_mrr)
        avg_sparse = sum(sparse_mrr) / len(sparse_mrr)
        avg_rrf = sum(rrf_mrr) / len(rrf_mrr)
        avg_final = sum(final_mrr) / len(final_mrr)

        global_totals["queries"] += len(tweets)
        global_totals["dense_sum"] += sum(dense_mrr)
        global_totals["sparse_sum"] += sum(sparse_mrr)
        global_totals["rrf_sum"] += sum(rrf_mrr)
        global_totals["final_sum"] += sum(final_mrr)

        global_results[lang] = {
            "Total Queries": len(tweets),
            "Dense MRR@5": avg_dense,
            "Sparse MRR@5": avg_sparse,
            "RRF MRR@5": avg_rrf,
            "Final MRR@5": avg_final,
        }

        print(f"\n--- Summary for {lang.upper()} ---")
        print(f"Total Queries:      {len(tweets)}\nDense MRR@5:        {avg_dense:.4f}")
        print(
            f"Sparse MRR@5:       {avg_sparse:.4f}\nRRF MRR@5:          {avg_rrf:.4f}\nFinal Pipeline:     {avg_final:.4f}"
        )

    # --- 4. PRINT BIG SUMMARY (UNCHANGED from original) ---
    print("\n\n" + "*" * 50)
    print("*" + " FINAL MULTILINGUAL EVALUATION SUMMARY ".center(48) + "*")
    print("*" * 50)

    for lang, metrics in global_results.items():
        print(f"\n[{lang.upper()}] - {metrics['Total Queries']} Queries Evaluated")
        print(
            f"  ├─ Dense Only:    {metrics['Dense MRR@5']:.4f}\n  ├─ Sparse Only:   {metrics['Sparse MRR@5']:.4f}"
        )
        print(
            f"  ├─ RRF Output:    {metrics['RRF MRR@5']:.4f}\n  └─ Final Rerank:  {metrics['Final MRR@5']:.4f}"
        )

    total_q = global_totals["queries"]
    print("\n==================================================")
    print(f"[GLOBAL AVERAGE] - {total_q} Total Queries Across All Languages")
    print(f"  ├─ Overall Dense:    {(global_totals['dense_sum'] / total_q):.4f}")
    print(f"  ├─ Overall Sparse:   {(global_totals['sparse_sum'] / total_q):.4f}")
    print(f"  ├─ Overall RRF:      {(global_totals['rrf_sum'] / total_q):.4f}")
    print(f"  └─ Overall Final:    {(global_totals['final_sum'] / total_q):.4f}")
    print("==================================================\n")

    # ==========================================
    # 5. XGBOOST FUSION: TRAIN & EVALUATE
    # ==========================================
    print("\n" + "=" * 50)
    print("  XGBOOST SCORE FUSION — TRAINING & EVALUATION")
    print("=" * 50)

    # Split at query level (80% train, 20% test)
    query_indices = list(range(len(all_xgb_data)))
    train_indices, test_indices = train_test_split(
        query_indices, test_size=0.2, random_state=42
    )

    # Build training data
    X_train_list, y_train_list = [], []
    for idx in train_indices:
        entry = all_xgb_data[idx]
        X_train_list.extend(entry["features"])
        y_train_list.extend(entry["labels"])

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)

    print(f"\nTraining set: {len(train_indices)} queries, {len(X_train)} samples")
    print(
        f"  Positive samples: {int(y_train.sum())} ({y_train.mean() * 100:.1f}%)"
    )
    print(f"Test set:     {len(test_indices)} queries")

    # Train the model
    score_fusion.train(X_train, y_train)

    # ==========================================
    # 6. EVALUATE XGBOOST ON TEST SPLIT
    # ==========================================
    print("\n" + "-" * 50)
    print("  XGBOOST TEST SET EVALUATION (20% Held-Out)")
    print("-" * 50)

    # Per-language tracking for fair comparison on the test split
    test_metrics = {
        lang: {
            "dense_mrrs": [],
            "sparse_mrrs": [],
            "rrf_mrrs": [],
            "rerank_mrrs": [],
            "xgb_mrrs": [],
        }
        for lang in languages
    }

    for idx in test_indices:
        entry = all_xgb_data[idx]
        lang = entry["lang"]
        true_pubkey = entry["true_pubkey"]

        # XGBoost prediction and reranking
        xgb_results = score_fusion.predict_and_rerank(
            entry["candidates"], entry["features"]
        )
        # Build doc_id → pubkey lookup for this query's candidates
        docid_to_pubkey = dict(
            zip(entry["candidates"], entry["candidate_pubkeys"])
        )
        xgb_preds = [docid_to_pubkey[doc_id] for doc_id, _ in xgb_results]
        xgb_mrr = MRR_at_5(xgb_preds, true_pubkey)

        test_metrics[lang]["dense_mrrs"].append(entry["dense_mrr"])
        test_metrics[lang]["sparse_mrrs"].append(entry["sparse_mrr"])
        test_metrics[lang]["rrf_mrrs"].append(entry["rrf_mrr"])
        test_metrics[lang]["rerank_mrrs"].append(entry["rerank_mrr"])
        test_metrics[lang]["xgb_mrrs"].append(xgb_mrr)

    # Compute and print test-split comparison
    all_dense, all_sparse, all_rrf, all_rerank, all_xgb = [], [], [], [], []

    for lang in languages:
        m = test_metrics[lang]
        n = len(m["xgb_mrrs"])
        if n == 0:
            continue

        avg_d = sum(m["dense_mrrs"]) / n
        avg_s = sum(m["sparse_mrrs"]) / n
        avg_r = sum(m["rrf_mrrs"]) / n
        avg_re = sum(m["rerank_mrrs"]) / n
        avg_x = sum(m["xgb_mrrs"]) / n

        all_dense.extend(m["dense_mrrs"])
        all_sparse.extend(m["sparse_mrrs"])
        all_rrf.extend(m["rrf_mrrs"])
        all_rerank.extend(m["rerank_mrrs"])
        all_xgb.extend(m["xgb_mrrs"])

        print(f"\n  [{lang.upper()}] — {n} Test Queries")
        print(f"    ├─ Dense Only:      {avg_d:.4f}")
        print(f"    ├─ Sparse Only:     {avg_s:.4f}")
        print(f"    ├─ RRF Output:      {avg_r:.4f}")
        print(f"    ├─ Reranker:        {avg_re:.4f}")
        print(f"    └─ XGBoost Fusion:  {avg_x:.4f}")

    # Global test-split averages
    n_test = len(all_xgb)
    print(f"\n  [GLOBAL TEST AVERAGE] — {n_test} Queries")
    print(f"    ├─ Dense Only:      {sum(all_dense) / n_test:.4f}")
    print(f"    ├─ Sparse Only:     {sum(all_sparse) / n_test:.4f}")
    print(f"    ├─ RRF Output:      {sum(all_rrf) / n_test:.4f}")
    print(f"    ├─ Reranker:        {sum(all_rerank) / n_test:.4f}")
    print(f"    └─ XGBoost Fusion:  {sum(all_xgb) / n_test:.4f}")
    print("=" * 50 + "\n")


@app.local_entrypoint()
def main():
    evaluate_pipeline.remote()
