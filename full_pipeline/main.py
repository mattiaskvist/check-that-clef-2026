import modal

from .retrievers import HarrierRetriever, SparseRetriever, BGEM3Retriever
from .utils import CHECKTHAT_DATASET, FusionProcessor, MRR_at_5, Recall_at_k, article_to_text
from .fusions import FeatureGenerator, RandomForestFuser

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
    gpu="A100-40GB",
    cpu=32.0,
    timeout=60 * 60 * 3,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={CACHE_MOUNT: embedding_cache},
)
def evaluate_pipeline():
    import numpy as np
    import pickle
    import os
    from datasets import load_dataset
    from sklearn.model_selection import KFold
    from tqdm import tqdm

    # --- CONFIGURABLE PARAMETERS ---
    CANDIDATE_TAKE_K = 500

    # --- LOAD BASE DATABASES (ALWAYS REQUIRED FOR EVAL) ---
    print("\nLoading dataset metadata...")
    collection_dataset = load_dataset(CHECKTHAT_DATASET, "collection", split="collection")
    article_pubkeys = [doc["pubkey"] for doc in collection_dataset]

    languages = ["de", "fr", "en"]
    lang_tweets = {}
    for lang in languages:
        lang_tweets[lang] = list(load_dataset(CHECKTHAT_DATASET, lang)["dev"])

    # --- CACHE CHECK ---
    cache_file = os.path.join(CACHE_MOUNT, f"cached_features_k{CANDIDATE_TAKE_K}.pkl")

    if os.path.exists(cache_file):
        print(f"\n>>> FOUND PERSISTENT EXTRACTED FEATURES IN MODAL VOLUME: {cache_file}")
        print(">>> SKIPPING RETRIEVAL INITIALIZATION. Loading direct to memory...")
        with open(cache_file, "rb") as f:
            cached_queries = pickle.load(f)
    else:
        # --- INITIALIZE COMPONENTS (HEAVY) ---
        print("\nInitialize Retrievers for fresh extraction...")
        dense_retriever = HarrierRetriever()
        sparse_retriever = SparseRetriever()
        fusion = FusionProcessor()

        # Load Heavy Texts
        article_texts = [article_to_text(doc) for doc in collection_dataset]
        dense_retriever.index(article_texts, cache_dir=CACHE_MOUNT)
        embedding_cache.commit()
        sparse_retriever.index(collection_dataset)

        # Pre-Encode Queries
        for lang in languages:
            query_texts = [row["text"] for row in lang_tweets[lang]]
            dense_retriever.index_queries(
                query_texts, cache_dir=CACHE_MOUNT, cache_name=f"queries_{lang}"
            )
            embedding_cache.commit()
        
        # Free GPU Memory
        dense_retriever.unload_model()

        # Extract
        print("\n" + "=" * 50)
        print("  PRE-COMPUTING FEATURES FOR ALL QUERIES")
        print("=" * 50)
        
        cached_queries = {lang: [] for lang in languages}

        for lang in languages:
            tweets = lang_tweets[lang]
            
            for i in tqdm(range(len(tweets)), desc=f"Extracting {lang.upper()}"):
                query_text = tweets[i]["text"]
                true_pubkey = tweets[i]["pubkey"]

                dense_ranks, dense_scores = dense_retriever.search(i, cache_name=f"queries_{lang}")
                sparse_ranks, sparse_scores = sparse_retriever.search(query_text)

                union_candidates = list(dict.fromkeys(dense_ranks[:CANDIDATE_TAKE_K] + sparse_ranks[:CANDIDATE_TAKE_K]))
                rrf_ranks, rrf_scores = fusion.reciprocal_rank_fusion([dense_ranks, sparse_ranks], top_k=len(union_candidates))
                
                features_rrf = FeatureGenerator.build_query_features(
                    union_candidates, dense_scores, dense_ranks, sparse_scores, sparse_ranks, rrf_scores, rrf_ranks, include_rrf=True
                )
                features_norrf = FeatureGenerator.build_query_features(
                    union_candidates, dense_scores, dense_ranks, sparse_scores, sparse_ranks, rrf_scores, rrf_ranks, include_rrf=False
                )
                
                labels = [1 if article_pubkeys[doc_id] == true_pubkey else 0 for doc_id in union_candidates]
                old_fused, _ = fusion.reciprocal_rank_fusion([dense_ranks, sparse_ranks], top_k=50)

                cached_queries[lang].append({
                    "true_pubkey": true_pubkey,
                    "candidates": union_candidates,
                    "features_rrf": features_rrf,
                    "features_norrf": features_norrf,
                    "labels": labels,
                    "b_dense": dense_ranks[:50],
                    "b_sparse": sparse_ranks[:50],
                    "b_rrf": old_fused[:50]
                })

        print("\nSaving extraction cache to persistent volume...")
        with open(cache_file, "wb") as f:
            pickle.dump(cached_queries, f)
        embedding_cache.commit()


    # --- RUN 5-FOLD CROSS VALIDATION ---
    print("\n" + "=" * 50)
    print("  RUNNING 5-FOLD CROSS VALIDATION")
    print("=" * 50)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = {}
    for lang in languages:
        splits[lang] = list(kf.split(range(len(lang_tweets[lang]))))

    model_configs = {}
    
    def add_config(name, est=100, depth=10, split=2, inc_rrf=True, is_global=False):
        model_configs[name] = {
            "fuser_class": RandomForestFuser,
            "global": is_global,
            "include_rrf": inc_rrf,
            "rf_params": {'n_estimators': est, 'max_depth': depth, 'min_samples_split': split}
        }
        
    # 1. Base Strategy
    add_config("RF [BASE] (est=100, d=10, s=2, RRF, PerLang)")
    
    # 2-3. Vary Trees
    add_config("RF [VARY EST] (est=50)", est=50)
    add_config("RF [VARY EST] (est=200)", est=200)
    
    # 4-5. Vary Depth
    add_config("RF [VARY DEPTH] (d=5)", depth=5)
    add_config("RF [VARY DEPTH] (d=15)", depth=15)
    
    # 6-7. Vary Node Split threshold
    add_config("RF [VARY SPLIT] (s=5)", split=5)
    add_config("RF [VARY SPLIT] (s=10)", split=10)
    
    # 8. Ablate External Feature
    add_config("RF [VARY FEAT] (No RRF)", inc_rrf=False)
    
    # 9. Ablate Language Consolidation
    add_config("RF [VARY SCOPE] (Global)", is_global=True)

    # Store 5 fold results
    results = {
        model_name: {lang: {"r10": [], "r30": [], "r50": []} for lang in languages}
        for model_name in model_configs.keys()
    }
    
    baselines_results = {
        lang: {"dense_r10": [], "dense_r30": [], "dense_r50": [], 
               "sparse_r10": [], "sparse_r30": [], "sparse_r50": [], 
               "rrf_r10": [], "rrf_r30": [], "rrf_r50": []}
        for lang in languages
    }

    for fold in range(5):
        print(f"\n--- FOLD {fold + 1} / 5 ---")
        
        for lang in languages:
            _, test_idx = splits[lang][fold]
            
            q_dense_r10, q_dense_r30, q_dense_r50 = [], [], []
            q_sparse_r10, q_sparse_r30, q_sparse_r50 = [], [], []
            q_rrf_r10, q_rrf_r30, q_rrf_r50 = [], [], []
            
            for i in test_idx:
                q_data = cached_queries[lang][i]
                pk = q_data["true_pubkey"]
                
                pks_dense = [article_pubkeys[idx] for idx in q_data["b_dense"]]
                q_dense_r10.append(Recall_at_k(pks_dense, pk, 10))
                q_dense_r30.append(Recall_at_k(pks_dense, pk, 30))
                q_dense_r50.append(Recall_at_k(pks_dense, pk, 50))
                
                pks_sp = [article_pubkeys[idx] for idx in q_data["b_sparse"]]
                q_sparse_r10.append(Recall_at_k(pks_sp, pk, 10))
                q_sparse_r30.append(Recall_at_k(pks_sp, pk, 30))
                q_sparse_r50.append(Recall_at_k(pks_sp, pk, 50))

                pks_rrf = [article_pubkeys[idx] for idx in q_data["b_rrf"]]
                q_rrf_r10.append(Recall_at_k(pks_rrf, pk, 10))
                q_rrf_r30.append(Recall_at_k(pks_rrf, pk, 30))
                q_rrf_r50.append(Recall_at_k(pks_rrf, pk, 50))
                
            baselines_results[lang]["dense_r10"].append(np.mean(q_dense_r10))
            baselines_results[lang]["dense_r30"].append(np.mean(q_dense_r30))
            baselines_results[lang]["dense_r50"].append(np.mean(q_dense_r50))
            baselines_results[lang]["sparse_r10"].append(np.mean(q_sparse_r10))
            baselines_results[lang]["sparse_r30"].append(np.mean(q_sparse_r30))
            baselines_results[lang]["sparse_r50"].append(np.mean(q_sparse_r50))
            baselines_results[lang]["rrf_r10"].append(np.mean(q_rrf_r10))
            baselines_results[lang]["rrf_r30"].append(np.mean(q_rrf_r30))
            baselines_results[lang]["rrf_r50"].append(np.mean(q_rrf_r50))

        # Model Evaluation
        print(f"  [Evaluating Baselines Complete]")
        for model_name, cfg in model_configs.items():
            print(f"  -> Training & Testing {model_name}...")
            inc_rrf = cfg["include_rrf"]
            feat_key = "features_rrf" if inc_rrf else "features_norrf"
            trained_models = {}

            if cfg["global"]:
                fuser = cfg["fuser_class"](include_rrf=inc_rrf, rf_params=cfg.get("rf_params", None))
                GX, Gy, Ggroup = [], [], []
                for lang in languages:
                    train_idx, _ = splits[lang][fold]
                    for i in train_idx:
                        q_data = cached_queries[lang][i]
                        GX.extend(q_data[feat_key])
                        Gy.extend(q_data["labels"])
                        Ggroup.append(len(q_data["candidates"]))
                
                fuser.train(
                    np.array(GX, dtype=np.float32), 
                    np.array(Gy, dtype=np.float32), 
                    np.array(Ggroup, dtype=np.int32)
                )
                for lang in languages:
                    trained_models[lang] = fuser
            else:
                for lang in languages:
                    fuser = cfg["fuser_class"](include_rrf=inc_rrf, rf_params=cfg.get("rf_params", None))
                    train_idx, _ = splits[lang][fold]
                    LX, Ly, Lgroup = [], [], []
                    for i in train_idx:
                        q_data = cached_queries[lang][i]
                        LX.extend(q_data[feat_key])
                        Ly.extend(q_data["labels"])
                        Lgroup.append(len(q_data["candidates"]))
                        
                    fuser.train(
                        np.array(LX, dtype=np.float32), 
                        np.array(Ly, dtype=np.float32), 
                        np.array(Lgroup, dtype=np.int32)
                    )
                    trained_models[lang] = fuser

            for lang in languages:
                _, test_idx = splits[lang][fold]
                fuser = trained_models[lang]
                
                q_r10, q_r30, q_r50 = [], [], []
                for i in test_idx:
                    q_data = cached_queries[lang][i]
                    feats = q_data[feat_key]
                    cands = q_data["candidates"]
                    true_pk = q_data["true_pubkey"]

                    reranked = fuser.predict_and_rerank(cands, feats)
                    top50 = [doc_id for doc_id, score in reranked[:50]]
                    top50_pks = [article_pubkeys[doc_id] for doc_id in top50]

                    q_r10.append(Recall_at_k(top50_pks, true_pk, 10))
                    q_r30.append(Recall_at_k(top50_pks, true_pk, 30))
                    q_r50.append(Recall_at_k(top50_pks, true_pk, 50))
                    
                results[model_name][lang]["r10"].append(np.mean(q_r10))
                results[model_name][lang]["r30"].append(np.mean(q_r30))
                results[model_name][lang]["r50"].append(np.mean(q_r50))


    # --- 6. PRINT MASSIVE SUMMARY POST-PROCESSING ---
    print("\n\n" + "*" * 60)
    print("*" + " FINAL MULTILINGUAL RECALL BENCHMARK ".center(58) + "*")
    print("*" * 60)

    for lang in languages:
        n_q = len(lang_tweets[lang])
        print(f"\n[{lang.upper()}] - {n_q} Queries (CV Mean ± Std)")
        b = baselines_results[lang]
        print(f"  [BASELINES]")
        print(f"  ├─ Dense Only:           R@10: {np.mean(b['dense_r10']):.4f} ± {np.std(b['dense_r10']):.4f}  |  R@30: {np.mean(b['dense_r30']):.4f} ± {np.std(b['dense_r30']):.4f}  |  R@50: {np.mean(b['dense_r50']):.4f} ± {np.std(b['dense_r50']):.4f}")
        print(f"  ├─ Sparse Only:          R@10: {np.mean(b['sparse_r10']):.4f} ± {np.std(b['sparse_r10']):.4f}  |  R@30: {np.mean(b['sparse_r30']):.4f} ± {np.std(b['sparse_r30']):.4f}  |  R@50: {np.mean(b['sparse_r50']):.4f} ± {np.std(b['sparse_r50']):.4f}")
        print(f"  └─ Static RRF:           R@10: {np.mean(b['rrf_r10']):.4f} ± {np.std(b['rrf_r10']):.4f}  |  R@30: {np.mean(b['rrf_r30']):.4f} ± {np.std(b['rrf_r30']):.4f}  |  R@50: {np.mean(b['rrf_r50']):.4f} ± {np.std(b['rrf_r50']):.4f}")
        
        print(f"  [ML FUSERS]")
        for model_name in sorted(model_configs.keys()):
            r10_mean, r10_std = np.mean(results[model_name][lang]["r10"]), np.std(results[model_name][lang]["r10"])
            r30_mean, r30_std = np.mean(results[model_name][lang]["r30"]), np.std(results[model_name][lang]["r30"])
            r50_mean, r50_std = np.mean(results[model_name][lang]["r50"]), np.std(results[model_name][lang]["r50"])
            prefix = "├─" if model_name != sorted(model_configs.keys())[-1] else "└─"
            print(f"  {prefix} {model_name:<44} R@10: {r10_mean:.4f} ± {r10_std:.4f}  |  R@30: {r30_mean:.4f} ± {r30_std:.4f}  |  R@50: {r50_mean:.4f} ± {r50_std:.4f}")

    print("\n" + "=" * 60)
    total_q = sum(len(lang_tweets[lang]) for lang in languages)
    print(f"[GLOBAL AVERAGE] - {total_q} Total Test Queries")

    global_b_dense_r10 = [np.average([baselines_results[l]["dense_r10"][f] for l in languages], weights=[len(splits[l][f][1]) for l in languages]) for f in range(5)]
    global_b_dense_r30 = [np.average([baselines_results[l]["dense_r30"][f] for l in languages], weights=[len(splits[l][f][1]) for l in languages]) for f in range(5)]
    global_b_dense_r50 = [np.average([baselines_results[l]["dense_r50"][f] for l in languages], weights=[len(splits[l][f][1]) for l in languages]) for f in range(5)]
    
    global_b_sparse_r10 = [np.average([baselines_results[l]["sparse_r10"][f] for l in languages], weights=[len(splits[l][f][1]) for l in languages]) for f in range(5)]
    global_b_sparse_r30 = [np.average([baselines_results[l]["sparse_r30"][f] for l in languages], weights=[len(splits[l][f][1]) for l in languages]) for f in range(5)]
    global_b_sparse_r50 = [np.average([baselines_results[l]["sparse_r50"][f] for l in languages], weights=[len(splits[l][f][1]) for l in languages]) for f in range(5)]
    
    global_b_rrf_r10 = [np.average([baselines_results[l]["rrf_r10"][f] for l in languages], weights=[len(splits[l][f][1]) for l in languages]) for f in range(5)]
    global_b_rrf_r30 = [np.average([baselines_results[l]["rrf_r30"][f] for l in languages], weights=[len(splits[l][f][1]) for l in languages]) for f in range(5)]
    global_b_rrf_r50 = [np.average([baselines_results[l]["rrf_r50"][f] for l in languages], weights=[len(splits[l][f][1]) for l in languages]) for f in range(5)]

    print(f"  [BASELINES]")
    print(f"  ├─ Dense Only:           R@10: {np.mean(global_b_dense_r10):.4f} ± {np.std(global_b_dense_r10):.4f}  |  R@30: {np.mean(global_b_dense_r30):.4f} ± {np.std(global_b_dense_r30):.4f}  |  R@50: {np.mean(global_b_dense_r50):.4f} ± {np.std(global_b_dense_r50):.4f}")
    print(f"  ├─ Sparse Only:          R@10: {np.mean(global_b_sparse_r10):.4f} ± {np.std(global_b_sparse_r10):.4f}  |  R@30: {np.mean(global_b_sparse_r30):.4f} ± {np.std(global_b_sparse_r30):.4f}  |  R@50: {np.mean(global_b_sparse_r50):.4f} ± {np.std(global_b_sparse_r50):.4f}")
    print(f"  └─ Static RRF:           R@10: {np.mean(global_b_rrf_r10):.4f} ± {np.std(global_b_rrf_r10):.4f}  |  R@30: {np.mean(global_b_rrf_r30):.4f} ± {np.std(global_b_rrf_r30):.4f}  |  R@50: {np.mean(global_b_rrf_r50):.4f} ± {np.std(global_b_rrf_r50):.4f}")

    print(f"  [ML FUSERS]")
    for model_name in sorted(model_configs.keys()):
        g_r10 = [np.average([results[model_name][l]["r10"][f] for l in languages], weights=[len(splits[l][f][1]) for l in languages]) for f in range(5)]
        g_r30 = [np.average([results[model_name][l]["r30"][f] for l in languages], weights=[len(splits[l][f][1]) for l in languages]) for f in range(5)]
        g_r50 = [np.average([results[model_name][l]["r50"][f] for l in languages], weights=[len(splits[l][f][1]) for l in languages]) for f in range(5)]
        
        prefix = "├─" if model_name != sorted(model_configs.keys())[-1] else "└─"
        print(f"  {prefix} {model_name:<44} R@10: {np.mean(g_r10):.4f} ± {np.std(g_r10):.4f}  |  R@30: {np.mean(g_r30):.4f} ± {np.std(g_r30):.4f}  |  R@50: {np.mean(g_r50):.4f} ± {np.std(g_r50):.4f}")

    print("==================================================\n")

@app.local_entrypoint()
def main():
    evaluate_pipeline.remote()
