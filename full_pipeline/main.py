import modal

from .retrievers import HarrierRetriever, SparseRetriever, BGEM3Retriever
from .utils import CHECKTHAT_DATASET, FusionProcessor, MRR_at_5, Recall_at_k, article_to_text
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
        "lightgbm",
        "scikit-learn",
        "country_converter",
        "geotext",
        "rapidfuzz",
        "spacy"
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
def evaluate_pipeline():
    import numpy as np
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    # --- INITIALIZE COMPONENTS ---
    dense_retriever = HarrierRetriever()
    sparse_retriever = SparseRetriever()
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
    languages = ["de", "fr", "en"]
    lang_tweets = {}
    for lang in languages:
        tweets = list(load_dataset(CHECKTHAT_DATASET, lang)["dev"])
        
        # Split into train/test per language
        train_idx, test_idx = train_test_split(list(range(len(tweets))), test_size=0.5, random_state=42)
        lang_tweets[lang] = {
            "tweets": tweets,
            "train_idx": train_idx,
            "test_idx": test_idx
        }
        
        query_texts = [row["text"] for row in tweets]
        dense_retriever.index_queries(
            query_texts, cache_dir=CACHE_MOUNT, cache_name=f"queries_{lang}"
        )
        embedding_cache.commit()

    dense_retriever.unload_model()

    # --- 4. TRAIN LIGHTGBM ---
    print("\n" + "=" * 50)
    print("  TRAINING LIGHTGBM SCORE FUSION (50% Data Per Language)")
    print("=" * 50)

    score_fusions = {}

    for lang in languages:
        X_train_list, y_train_list, group_train_list = [], [], []
        tweets = lang_tweets[lang]["tweets"]
        
        for i in tqdm(lang_tweets[lang]["train_idx"], desc=f"Training features for {lang.upper()}"):
            query_text = tweets[i]["text"]
            true_pubkey = tweets[i]["pubkey"]

            dense_ranks, dense_scores = dense_retriever.search(i, cache_name=f"queries_{lang}")
            sparse_ranks, sparse_scores = sparse_retriever.search(query_text)

            # Candidate Union (top 100 dense + top 100 sparse)
            union_candidates = list(dict.fromkeys(dense_ranks[:100] + sparse_ranks[:100]))

            rrf_ranks, rrf_scores = fusion.reciprocal_rank_fusion([dense_ranks, sparse_ranks], top_k=len(union_candidates))
            
            features = ScoreFusionProcessor.build_query_features(
                candidates=union_candidates,
                dense_scores=dense_scores,
                dense_ranked=dense_ranks,
                sparse_scores=sparse_scores,
                sparse_ranked=sparse_ranks,
                rrf_scores=rrf_scores,
                rrf_ranked=rrf_ranks,
            )

            labels = [1 if article_pubkeys[doc_id] == true_pubkey else 0 for doc_id in union_candidates]
            
            X_train_list.extend(features)
            y_train_list.extend(labels)
            group_train_list.append(len(union_candidates))

        X_train = np.array(X_train_list, dtype=np.float32)
        y_train = np.array(y_train_list, dtype=np.float32)
        group_train = np.array(group_train_list, dtype=np.int32)
        
        print(f"\n[{lang.upper()}] Training set: {len(X_train)} samples")
        print(f"  Positive samples: {int(y_train.sum())} ({y_train.mean() * 100:.1f}%)")
        
        score_fusions[lang] = ScoreFusionProcessor()
        score_fusions[lang].train(X_train, y_train, group=group_train)

    # --- 5. TEST PIPELINE (50% Data) ---
    print("\n" + "=" * 50)
    print("  TESTING PIPELINE (50% Held-Out)")
    print("=" * 50)

    test_metrics = {lang: {"dense_r10": [], "dense_r30": [], "dense_r50": [], "sparse_r10": [], "sparse_r30": [], "sparse_r50": [], "rrf_r10": [], "rrf_r30": [], "rrf_r50": [], "lgb_r10": [], "lgb_r30": [], "lgb_r50": []} for lang in languages}

    for lang in languages:
        tweets = lang_tweets[lang]["tweets"]
        for i in tqdm(lang_tweets[lang]["test_idx"], desc=f"Testing {lang.upper()}"):
            query_text = tweets[i]["text"]
            true_pubkey = tweets[i]["pubkey"]

            dense_ranks, dense_scores = dense_retriever.search(i, cache_name=f"queries_{lang}")
            sparse_ranks, sparse_scores = sparse_retriever.search(query_text)

            test_metrics[lang]["dense_r10"].append(Recall_at_k([article_pubkeys[idx] for idx in dense_ranks], true_pubkey, 10))
            test_metrics[lang]["dense_r30"].append(Recall_at_k([article_pubkeys[idx] for idx in dense_ranks], true_pubkey, 30))
            test_metrics[lang]["dense_r50"].append(Recall_at_k([article_pubkeys[idx] for idx in dense_ranks], true_pubkey, 50))
            
            test_metrics[lang]["sparse_r10"].append(Recall_at_k([article_pubkeys[idx] for idx in sparse_ranks], true_pubkey, 10))
            test_metrics[lang]["sparse_r30"].append(Recall_at_k([article_pubkeys[idx] for idx in sparse_ranks], true_pubkey, 30))
            test_metrics[lang]["sparse_r50"].append(Recall_at_k([article_pubkeys[idx] for idx in sparse_ranks], true_pubkey, 50))

            # Old Pipeline Fuser: RRF Top 50
            old_fused, old_rrf_scores = fusion.reciprocal_rank_fusion([dense_ranks, sparse_ranks], top_k=50)
            test_metrics[lang]["rrf_r10"].append(Recall_at_k([article_pubkeys[idx] for idx in old_fused], true_pubkey, 10))
            test_metrics[lang]["rrf_r30"].append(Recall_at_k([article_pubkeys[idx] for idx in old_fused], true_pubkey, 30))
            test_metrics[lang]["rrf_r50"].append(Recall_at_k([article_pubkeys[idx] for idx in old_fused], true_pubkey, 50))
            
            # New Pipeline Fuser: LGB
            union_candidates = list(dict.fromkeys(dense_ranks[:100] + sparse_ranks[:100]))
            rrf_ranks, rrf_scores = fusion.reciprocal_rank_fusion([dense_ranks, sparse_ranks], top_k=len(union_candidates))
            
            features = ScoreFusionProcessor.build_query_features(
                candidates=union_candidates,
                dense_scores=dense_scores,
                dense_ranked=dense_ranks,
                sparse_scores=sparse_scores,
                sparse_ranked=sparse_ranks,
                rrf_scores=rrf_scores,
                rrf_ranked=rrf_ranks,
            )

            lgb_results = score_fusions[lang].predict_and_rerank(union_candidates, features)
            lgb_top50 = [doc_id for doc_id, score in lgb_results[:50]]
            test_metrics[lang]["lgb_r10"].append(Recall_at_k([article_pubkeys[idx] for idx in lgb_top50], true_pubkey, 10))
            test_metrics[lang]["lgb_r30"].append(Recall_at_k([article_pubkeys[idx] for idx in lgb_top50], true_pubkey, 30))
            test_metrics[lang]["lgb_r50"].append(Recall_at_k([article_pubkeys[idx] for idx in lgb_top50], true_pubkey, 50))

    # Calculate global testing metrics
    print("\n\n" + "*" * 50)
    print("*" + " FINAL MULTILINGUAL RECALL TEST ".center(48) + "*")
    print("*" * 50)

    global_test = {k: 0.0 for k in test_metrics["en"].keys()}
    total_q = 0

    for lang in languages:
        m = test_metrics[lang]
        n_q = len(m["dense_r10"])
        if n_q == 0: continue
        total_q += n_q
        print(f"\n[{lang.upper()}] - {n_q} Queries")
        print(f"  ├─ Dense Only:        R@10: {sum(m['dense_r10'])/n_q:.4f}  |  R@30: {sum(m['dense_r30'])/n_q:.4f}  |  R@50: {sum(m['dense_r50'])/n_q:.4f}")
        print(f"  ├─ Sparse Only:       R@10: {sum(m['sparse_r10'])/n_q:.4f}  |  R@30: {sum(m['sparse_r30'])/n_q:.4f}  |  R@50: {sum(m['sparse_r50'])/n_q:.4f}")
        print(f"  ├─ RRF Fusion:        R@10: {sum(m['rrf_r10'])/n_q:.4f}  |  R@30: {sum(m['rrf_r30'])/n_q:.4f}  |  R@50: {sum(m['rrf_r50'])/n_q:.4f}")
        print(f"  └─ LGB Fusion:        R@10: {sum(m['lgb_r10'])/n_q:.4f}  |  R@30: {sum(m['lgb_r30'])/n_q:.4f}  |  R@50: {sum(m['lgb_r50'])/n_q:.4f}")

        for k in global_test.keys():
            global_test[k] += sum(m[k])

    print("\n==================================================")
    print(f"[GLOBAL AVERAGE] - {total_q} Total Test Queries")
    print(f"  ├─ Dense Only:        R@10: {global_test['dense_r10']/total_q:.4f}  |  R@30: {global_test['dense_r30']/total_q:.4f}  |  R@50: {global_test['dense_r50']/total_q:.4f}")
    print(f"  ├─ Sparse Only:       R@10: {global_test['sparse_r10']/total_q:.4f}  |  R@30: {global_test['sparse_r30']/total_q:.4f}  |  R@50: {global_test['sparse_r50']/total_q:.4f}")
    print(f"  ├─ RRF Fusion:        R@10: {global_test['rrf_r10']/total_q:.4f}  |  R@30: {global_test['rrf_r30']/total_q:.4f}  |  R@50: {global_test['rrf_r50']/total_q:.4f}")
    print(f"  └─ LGB Fusion:        R@10: {global_test['lgb_r10']/total_q:.4f}  |  R@30: {global_test['lgb_r30']/total_q:.4f}  |  R@50: {global_test['lgb_r50']/total_q:.4f}")
    print("==================================================\n")

@app.local_entrypoint()
def main():
    evaluate_pipeline.remote()
