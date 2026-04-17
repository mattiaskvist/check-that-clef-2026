import modal

from .fusions import RRFFuser, RandomForestFuser
from .rerankers import NemotronReranker
from .retrievers import HarrierRetriever, SparseRetriever
from .submission import (
    SUBMISSION_TOP_K,
    modal_volume_download_command,
    normalize_split,
    submission_volume_remote_dir,
    write_submission_tsv_files,
)
from .utils import (
    CHECKTHAT_DATASET,
    MRR_at_5,
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
        "scikit-learn",
    )
)

app = modal.App("checkthat-evaluation-pipeline")
EMBEDDING_CACHE_VOLUME_NAME = "checkthat-embedding-cache"
embedding_cache = modal.Volume.from_name(
    EMBEDDING_CACHE_VOLUME_NAME, create_if_missing=True
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
def evaluate_pipeline(
    force_recompute_sparse_cache: bool = False,
    force_recompute_dense_documents: bool = False,
    force_recompute_dense_queries: bool = False,
    split: str = "dev",
    collect_submission: bool = False,
    submission_volume_subdir: str = "submissions",
    fusion_method: str = "rrf",
    force_retrain_fusion: bool = False,
    global_fusion_model: bool = False,
):
    from datetime import datetime, timezone

    from datasets import load_dataset
    from tqdm import tqdm

    # --- INITIALIZE COMPONENTS ---
    split = normalize_split(split)

    dense_retriever = HarrierRetriever()
    sparse_retriever = SparseRetriever()
    reranker = NemotronReranker()
    FUSION_TOP_K = 30  # Number of candidates to fuse and rerank
    SPARSE_CACHE_TOP_K = (
        2000  # Keep a deep sparse candidate pool for fast rerank/fusion iteration
    )

    # --- INITIALIZE FUSER ---
    if fusion_method == "random_forest":
        fuser = RandomForestFuser(global_model=global_fusion_model)
    else:
        fuser = RRFFuser()

    # --- LOAD & INDEX COLLECTION ---
    collection_dataset = load_dataset(
        CHECKTHAT_DATASET, "collection", split="collection"
    )
    collection_documents = collection_dataset.to_list()
    article_texts = [
        reranker.document_to_text(doc)[: reranker.max_length] # Truncate to avoid memory issues 
        for doc in collection_documents
    ]
    article_pubkeys = [doc["pubkey"] for doc in collection_documents]
    reranker_corpus = reranker.preprocess_corpus(collection_documents)

    dense_retriever.index(
        article_texts,
        cache_dir=CACHE_MOUNT,
        force_recompute=force_recompute_dense_documents,
    )
    embedding_cache.commit()
    sparse_retriever.index(collection_dataset)

    # --- 3. PRE-ENCODE ALL QUERIES, THEN FREE EMBEDDING MODEL ---
    languages = ["de", "fr", "en"]
    lang_tweets = {}
    for lang in languages:
        tweets = list(load_dataset(CHECKTHAT_DATASET, lang)[split])
        lang_tweets[lang] = tweets
        query_texts = [row["text"] for row in tweets]
        dense_retriever.index_queries(
            query_texts,
            cache_dir=CACHE_MOUNT,
            cache_name=f"queries_{lang}",
            force_recompute=force_recompute_dense_queries,
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

    # --- 3b. RF FUSION TRAINING (if applicable) ---
    if isinstance(fuser, RandomForestFuser):
        sparse_config = {
            "k1": sparse_retriever.bm25_k1,
            "b": sparse_retriever.bm25_b,
            "stemmer": "lancaster",
        }
        loaded = False
        if not force_retrain_fusion:
            loaded = fuser.load(
                cache_dir=CACHE_MOUNT,
                dense_model_name=dense_retriever.model_name,
                sparse_config=sparse_config,
                train_split="train",
            )
        if not loaded:
            # Load and pre-encode train queries
            train_tweets_by_lang = {}
            for lang in languages:
                train_tweets = list(
                    load_dataset(CHECKTHAT_DATASET, lang)["train"]
                )
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
                    top_k=SPARSE_CACHE_TOP_K,
                    force_recompute=force_recompute_sparse_cache,
                )
                embedding_cache.commit()

            fuser.train(
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
                train_tweets_by_lang=train_tweets_by_lang,
                article_pubkeys=article_pubkeys,
                dense_model_name=dense_retriever.model_name,
                sparse_config=sparse_config,
            )
            fuser.save(cache_dir=CACHE_MOUNT)
            embedding_cache.commit()

    dense_retriever.unload_model()

    # --- 4. EVALUATION LOOP ---
    global_results = {}
    submission_predictions = {lang: [] for lang in languages} if collect_submission else {}

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
        has_labels = bool(tweets) and "pubkey" in tweets[0]

        # Trackers for the current language
        lang_metrics = (
            {
                "dense": {m: [] for m in ["mrr5", "r5", "r10", "r30", "r50"]},
                "sparse": {m: [] for m in ["mrr5", "r5", "r10", "r30", "r50"]},
                "rrf": {"mrr5": [], f"r{FUSION_TOP_K}": []},
                "final": {"mrr5": [], "r5": []},
            }
            if has_labels
            else None
        )

        for i, row in enumerate(
            tqdm(tweets, desc=f"Evaluating {lang.upper()} Queries")
        ):
            query_text = row["text"]
            true_pubkey = row.get("pubkey")

            # Step A: Independent Retrieval
            if isinstance(fuser, RandomForestFuser):
                dense_ranks, dense_scores = dense_retriever.search_with_scores(
                    i, cache_name=f"queries_{lang}"
                )
                sparse_ranks, sparse_scores = sparse_retriever.search_with_scores(
                    i, cache_name=f"sparse_queries_{lang}"
                )
            else:
                dense_ranks = dense_retriever.search(i, cache_name=f"queries_{lang}")
                sparse_ranks = sparse_retriever.search(
                    i, cache_name=f"sparse_queries_{lang}"
                )
                dense_scores = None
                sparse_scores = None

            dense_preds = [article_pubkeys[doc_id] for doc_id in dense_ranks]
            sparse_preds = [article_pubkeys[doc_id] for doc_id in sparse_ranks]

            if has_labels:
                # Dense Metrics
                lang_metrics["dense"]["mrr5"].append(MRR_at_5(dense_preds, true_pubkey))
                lang_metrics["dense"]["r5"].append(
                    recall_at_K(dense_preds, true_pubkey, 5)
                )
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
            fused_candidates = fuser.fuse(
                ranked_lists=[dense_ranks, sparse_ranks],
                scores_lists=(
                    [dense_scores, sparse_scores] if dense_scores is not None else None
                ),
                top_k=FUSION_TOP_K,
                lang=lang,
            )
            rrf_preds = [article_pubkeys[doc_id] for doc_id in fused_candidates]

            if has_labels:
                lang_metrics["rrf"]["mrr5"].append(MRR_at_5(rrf_preds, true_pubkey))
                lang_metrics["rrf"][f"r{FUSION_TOP_K}"].append(
                    recall_at_K(rrf_preds, true_pubkey, FUSION_TOP_K)
                )

            # Step C: Reranking
            final_results = reranker.rerank(
                query=query_text,
                doc_indices=fused_candidates,
                corpus=reranker_corpus,
            )
            final_preds = [article_pubkeys[doc_id] for doc_id, score in final_results]

            if collect_submission:
                submission_predictions[lang].append(
                    {
                        "index": row.get("index", i),
                        "preds": final_preds[:SUBMISSION_TOP_K],
                    }
                )

            if has_labels:
                lang_metrics["final"]["mrr5"].append(MRR_at_5(final_preds, true_pubkey))
                lang_metrics["final"]["r5"].append(
                    recall_at_K(final_preds, true_pubkey, 5)
                )

        # Track and print metrics for the current language
        num_queries = len(tweets)
        if has_labels:
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
        else:
            global_results[lang] = {"Total Queries": num_queries, "metrics": None}
            print(
                f"\n--- Summary for {lang.upper()} ({num_queries} Queries) ---\nNo labels available for split '{split}'; generated predictions only."
            )

    # --- 4. PRINT BIG SUMMARY ---
    if global_totals["queries"] > 0:
        print("\n\n" + "*" * 80)
        print("*" + " FINAL MULTILINGUAL EVALUATION SUMMARY ".center(78) + "*")
        print("*" * 80)

        for lang, data in global_results.items():
            m = data["metrics"]
            if m is None:
                print(f"\n[{lang.upper()}] - {data['Total Queries']} Queries (No labels)")
                continue

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

    submission_artifacts = None
    if collect_submission:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
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
    force_retrain_fusion: bool = False,
    global_fusion_model: bool = False,
):
    run_output = evaluate_pipeline.remote(
        force_recompute_sparse_cache=force_recompute_sparse_cache,
        force_recompute_dense_documents=force_recompute_dense_documents,
        force_recompute_dense_queries=force_recompute_dense_queries,
        split=split,
        collect_submission=export_submission_tsv,
        submission_volume_subdir=submission_volume_subdir,
        fusion_method=fusion_method,
        force_retrain_fusion=force_retrain_fusion,
        global_fusion_model=global_fusion_model,
    )

    if export_submission_tsv:
        submission_artifacts = run_output["submission_artifacts"]
        download_command = modal_volume_download_command(
            volume_name=submission_artifacts["volume_name"],
            remote_dir=submission_artifacts["remote_dir"],
            local_destination=submission_download_dir,
        )
        print("Wrote submission files to Modal volume:")
        print(
            f"  - Volume: {submission_artifacts['volume_name']}"
        )
        print(f"  - Remote dir: {submission_artifacts['remote_dir']}")
        print("  - Files:")
        for filename in submission_artifacts["written_files"]:
            print(f"    - {filename}")
        print("\nRun this command to download them locally:")
        print(f"  {download_command}")
