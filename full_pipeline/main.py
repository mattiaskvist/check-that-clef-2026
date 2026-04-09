import modal

from .rerankers import Gemma2BReranker, NemotronReranker
from .retrievers import BGEM3Retriever, HarrierRetriever, SparseRetriever
from .utils import CHECKTHAT_DATASET, FusionProcessor, MRR_at_5, article_to_text

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
embedding_cache = modal.Volume.from_name("checkthat-embedding-cache", create_if_missing=True)

CACHE_MOUNT = "/cache/embeddings"


# ==========================================
# 2. MAIN CLOUD FUNCTION
# ==========================================
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={CACHE_MOUNT: embedding_cache},
)
def evaluate_pipeline():
    from datasets import load_dataset
    from tqdm import tqdm

    # --- INITIALIZE COMPONENTS ---
    dense_retriever = HarrierRetriever()
    sparse_retriever = SparseRetriever()
    reranker = NemotronReranker()
    fusion = FusionProcessor()

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
    languages = ["fr"]
    lang_tweets = {}
    for lang in languages:
        tweets = list(load_dataset(CHECKTHAT_DATASET, lang)["dev"])
        lang_tweets[lang] = tweets
        query_texts = [row["text"] for row in tweets]
        dense_retriever.index_queries(query_texts, cache_dir=CACHE_MOUNT, cache_name=f"queries_{lang}")
        embedding_cache.commit()

    dense_retriever.unload_model()

    # --- 4. EVALUATION LOOP ---
    global_results = {}
    global_totals = {
        "queries": 0,
        "dense_sum": 0.0,
        "sparse_sum": 0.0,
        "rrf_sum": 0.0,
        "final_sum": 0.0,
    }

    for lang in languages:
        print("\n==========================================")
        print(f"  STARTING EVALUATION FOR LANGUAGE: {lang.upper()}")
        print("==========================================")

        tweets = lang_tweets[lang]

        dense_mrr, sparse_mrr, rrf_mrr, final_mrr = [], [], [], []

        for i, row in enumerate(tqdm(tweets, desc=f"Evaluating {lang.upper()} Queries")):
            query_text = row["text"]
            true_pubkey = row["pubkey"]

            # Step A: Independent Retrieval
            dense_ranks = dense_retriever.search(i)
            sparse_ranks = sparse_retriever.search(query_text)

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

            # Step B: Fusion
            fused_candidates = fusion.reciprocal_rank_fusion(
                [dense_ranks, sparse_ranks], top_k=20
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

        # Track and print metrics
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

    # --- 4. PRINT BIG SUMMARY ---
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


@app.local_entrypoint()
def main():
    evaluate_pipeline.remote()
