import modal

from .rerankers import NemotronReranker
from .retrievers import HarrierRetriever, SparseRetriever
from .utils import (
    CHECKTHAT_DATASET,
    FusionProcessor,
    MRR_at_5,
    article_to_text,
    recall_at_K,
)

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
    gpu="A100-80GB",
    timeout=60 * 60 * 3,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={CACHE_MOUNT: embedding_cache},
)
def evaluate_pipeline(force_recompute_sparse_cache: bool = False):
    from datasets import load_dataset
    from tqdm import tqdm

    # --- INITIALIZE COMPONENTS ---
    dense_retriever = HarrierRetriever()
    sparse_retriever = SparseRetriever()
    reranker = NemotronReranker()
    fusion = FusionProcessor()
    FUSION_TOP_K = 30  # Number of candidates to fuse and rerank
    SPARSE_CACHE_TOP_K = (
        2000  # Keep a deep sparse candidate pool for fast rerank/fusion iteration
    )

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
        sparse_retriever.index_queries(
            query_texts,
            lang=lang,
            cache_dir=CACHE_MOUNT,
            cache_name=f"sparse_queries_{lang}",
            top_k=SPARSE_CACHE_TOP_K,
            force_recompute=force_recompute_sparse_cache,
        )
        embedding_cache.commit()

    dense_retriever.unload_model()

    # --- 4. EVALUATION LOOP ---
    global_results = {}

    # Initialize global tracking dictionaries
    global_totals = {
        "queries": 0,
        "dense": {m: 0.0 for m in ["mrr5", "r5", "r10", "r30", "r50"]},
        "sparse": {m: 0.0 for m in ["mrr5", "r5", "r10", "r30", "r50"]},
        "rrf": {"mrr5": 0.0, f"r{FUSION_TOP_K}": 0.0},
        "final": {"mrr5": 0.0, "r5": 0.0},
    }

    for lang in languages:
        print("\n==========================================")
        print(f"  STARTING EVALUATION FOR LANGUAGE: {lang.upper()}")
        print("==========================================")

        tweets = lang_tweets[lang]

        # Trackers for the current language
        lang_metrics = {
            "dense": {m: [] for m in ["mrr5", "r5", "r10", "r30", "r50"]},
            "sparse": {m: [] for m in ["mrr5", "r5", "r10", "r30", "r50"]},
            "rrf": {"mrr5": [], f"r{FUSION_TOP_K}": []},
            "final": {"mrr5": [], "r5": []},
        }

        for i, row in enumerate(
            tqdm(tweets, desc=f"Evaluating {lang.upper()} Queries")
        ):
            query_text = row["text"]
            true_pubkey = row["pubkey"]

            # Step A: Independent Retrieval
            dense_ranks = dense_retriever.search(i, cache_name=f"queries_{lang}")
            sparse_ranks = sparse_retriever.search(
                i, cache_name=f"sparse_queries_{lang}"
            )

            dense_preds = [article_pubkeys[doc_id] for doc_id in dense_ranks]
            sparse_preds = [article_pubkeys[doc_id] for doc_id in sparse_ranks]

            # Dense Metrics
            lang_metrics["dense"]["mrr5"].append(MRR_at_5(dense_preds, true_pubkey))
            lang_metrics["dense"]["r5"].append(recall_at_K(dense_preds, true_pubkey, 5))
            lang_metrics["dense"]["r10"].append(
                recall_at_K(dense_preds, true_pubkey, 10)
            )
            lang_metrics["dense"]["r30"].append(
                recall_at_K(dense_preds, true_pubkey, 30)
            )
            lang_metrics["dense"]["r50"].append(
                recall_at_K(dense_preds, true_pubkey, 50)
            )

            # Sparse Metrics
            lang_metrics["sparse"]["mrr5"].append(MRR_at_5(sparse_preds, true_pubkey))
            lang_metrics["sparse"]["r5"].append(
                recall_at_K(sparse_preds, true_pubkey, 5)
            )
            lang_metrics["sparse"]["r10"].append(
                recall_at_K(sparse_preds, true_pubkey, 10)
            )
            lang_metrics["sparse"]["r30"].append(
                recall_at_K(sparse_preds, true_pubkey, 30)
            )
            lang_metrics["sparse"]["r50"].append(
                recall_at_K(sparse_preds, true_pubkey, 50)
            )

            # Step B: Fusion
            fused_candidates = fusion.reciprocal_rank_fusion(
                [dense_ranks, sparse_ranks], top_k=FUSION_TOP_K
            )
            rrf_preds = [article_pubkeys[doc_id] for doc_id in fused_candidates]

            lang_metrics["rrf"]["mrr5"].append(MRR_at_5(rrf_preds, true_pubkey))
            lang_metrics["rrf"][f"r{FUSION_TOP_K}"].append(
                recall_at_K(rrf_preds, true_pubkey, FUSION_TOP_K)
            )

            # Step C: Reranking
            final_results = reranker.rerank(query_text, fused_candidates, article_texts)
            final_preds = [article_pubkeys[doc_id] for doc_id, score in final_results]

            lang_metrics["final"]["mrr5"].append(MRR_at_5(final_preds, true_pubkey))
            lang_metrics["final"]["r5"].append(recall_at_K(final_preds, true_pubkey, 5))

        # Track and print metrics for the current language
        num_queries = len(tweets)
        global_totals["queries"] += num_queries

        # Calculate averages for current language and add to global sums
        avg_metrics = {}
        for stage, metrics in lang_metrics.items():
            avg_metrics[stage] = {}
            for metric_name, values in metrics.items():
                avg_val = sum(values) / num_queries
                avg_metrics[stage][metric_name] = avg_val
                global_totals[stage][metric_name] += sum(values)

        global_results[lang] = {"Total Queries": num_queries, "metrics": avg_metrics}

        print(f"\n--- Summary for {lang.upper()} ({num_queries} Queries) ---")
        print(
            f"Dense Only:   MRR@5: {avg_metrics['dense']['mrr5']:.4f} | R@5: {avg_metrics['dense']['r5']:.4f} | R@10: {avg_metrics['dense']['r10']:.4f} | R@30: {avg_metrics['dense']['r30']:.4f} | R@50: {avg_metrics['dense']['r50']:.4f}"
        )
        print(
            f"Sparse Only:  MRR@5: {avg_metrics['sparse']['mrr5']:.4f} | R@5: {avg_metrics['sparse']['r5']:.4f} | R@10: {avg_metrics['sparse']['r10']:.4f} | R@30: {avg_metrics['sparse']['r30']:.4f} | R@50: {avg_metrics['sparse']['r50']:.4f}"
        )
        print(
            f"RRF Output:   MRR@5: {avg_metrics['rrf']['mrr5']:.4f} | R@{FUSION_TOP_K}: {avg_metrics['rrf'][f'r{FUSION_TOP_K}']:.4f}"
        )
        print(
            f"Final Rerank: MRR@5: {avg_metrics['final']['mrr5']:.4f} | R@5: {avg_metrics['final']['r5']:.4f}"
        )

    # --- 4. PRINT BIG SUMMARY ---
    print("\n\n" + "*" * 80)
    print("*" + " FINAL MULTILINGUAL EVALUATION SUMMARY ".center(78) + "*")
    print("*" * 80)

    for lang, data in global_results.items():
        m = data["metrics"]
        print(f"\n[{lang.upper()}] - {data['Total Queries']} Queries Evaluated")
        print(
            f"  ├─ Dense Only:    MRR@5: {m['dense']['mrr5']:.4f} | R@5: {m['dense']['r5']:.4f} | R@10: {m['dense']['r10']:.4f} | R@30: {m['dense']['r30']:.4f} | R@50: {m['dense']['r50']:.4f}"
        )
        print(
            f"  ├─ Sparse Only:   MRR@5: {m['sparse']['mrr5']:.4f} | R@5: {m['sparse']['r5']:.4f} | R@10: {m['sparse']['r10']:.4f} | R@30: {m['sparse']['r30']:.4f} | R@50: {m['sparse']['r50']:.4f}"
        )
        print(
            f"  ├─ RRF Output:    MRR@5: {m['rrf']['mrr5']:.4f} | R@{FUSION_TOP_K}: {m['rrf'][f'r{FUSION_TOP_K}']:.4f}"
        )
        print(
            f"  └─ Final Rerank:  MRR@5: {m['final']['mrr5']:.4f} | R@5: {m['final']['r5']:.4f}"
        )

    total_q = global_totals["queries"]

    # Calculate global averages
    g_avg = {
        stage: {m: global_totals[stage][m] / total_q for m in global_totals[stage]}
        for stage in ["dense", "sparse", "rrf", "final"]
    }

    print(
        "\n================================================================================"
    )
    print(f"[GLOBAL AVERAGE] - {total_q} Total Queries Across All Languages")
    print(
        f"  ├─ Overall Dense:    MRR@5: {g_avg['dense']['mrr5']:.4f} | R@5: {g_avg['dense']['r5']:.4f} | R@10: {g_avg['dense']['r10']:.4f} | R@30: {g_avg['dense']['r30']:.4f} | R@50: {g_avg['dense']['r50']:.4f}"
    )
    print(
        f"  ├─ Overall Sparse:   MRR@5: {g_avg['sparse']['mrr5']:.4f} | R@5: {g_avg['sparse']['r5']:.4f} | R@10: {g_avg['sparse']['r10']:.4f} | R@30: {g_avg['sparse']['r30']:.4f} | R@50: {g_avg['sparse']['r50']:.4f}"
    )
    print(
        f"  ├─ Overall RRF:      MRR@5: {g_avg['rrf']['mrr5']:.4f} | R@{FUSION_TOP_K}: {g_avg['rrf'][f'r{FUSION_TOP_K}']:.4f}"
    )
    print(
        f"  └─ Overall Final:    MRR@5: {g_avg['final']['mrr5']:.4f} | R@5: {g_avg['final']['r5']:.4f}"
    )
    print(
        "================================================================================\n"
    )


@app.local_entrypoint()
def main(force_recompute_sparse_cache: bool = False):
    evaluate_pipeline.remote(force_recompute_sparse_cache=force_recompute_sparse_cache)
