import modal

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
    )
)

app = modal.App("checkthat-evaluation-pipeline")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
# (Modal automatically serializes these and sends them to the container)


def article_to_text(doc: dict) -> str:
    return f"{doc['title']}\n{doc['abstract']}"


def MRR_at_5(preds, label) -> float:
    preds = preds[:5]
    if label in preds:
        return 1 / (preds.index(label) + 1)
    else:
        return 0.0


def last_logit_pool(logits, attention_mask):
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return logits[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return torch.stack(
            [logits[i, sequence_lengths[i]] for i in range(batch_size)], dim=0
        )


def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
    if prompt is None:
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    sep = "\n"
    prompt_inputs = tokenizer(prompt, return_tensors=None, add_special_tokens=False)[
        "input_ids"
    ]
    sep_inputs = tokenizer(sep, return_tensors=None, add_special_tokens=False)[
        "input_ids"
    ]

    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(
            f"Query A: {query}",
            return_tensors=None,
            add_special_tokens=False,
            max_length=max_length * 3 // 4,
            truncation=True,
        )
        passage_inputs = tokenizer(
            f"Passage B: {passage}\nAnswer:",
            return_tensors=None,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )

        # --- Bypass prepare_for_model using manual concatenation ---
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        q_ids = bos + query_inputs["input_ids"]
        p_ids = sep_inputs + passage_inputs["input_ids"]

        if len(q_ids) + len(p_ids) > max_length:
            p_ids = p_ids[: max_length - len(q_ids)]

        item = {"input_ids": q_ids + p_ids + sep_inputs + prompt_inputs}
        item["attention_mask"] = [1] * len(item["input_ids"])
        # ----------------------------------------------------------------

        inputs.append(item)

    return tokenizer.pad(
        inputs,
        padding=True,
        max_length=max_length + len(sep_inputs) + len(prompt_inputs),
        pad_to_multiple_of=8,
        return_tensors="pt",
    )


# ==========================================
# 3. MAIN CLOUD FUNCTION
# ==========================================
@app.function(
    image=image,
    gpu="A100-40GB",  # A100-40 (40GB VRAM) is perfect and cost-effective for a 2B LLM
    timeout=60 * 60 * 2,  # 2 hours
    secrets=[
        modal.Secret.from_name("hf-token")
    ],  # Pulls your HF token to access the private repo
)
def evaluate_pipeline():
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer, util
    from peft import PeftModel
    from rank_bm25 import BM25Okapi
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- 1. LOAD MODELS (ONCE) ---
    print("Loading Dense Retriever (BGE-M3 + LoRA)...")
    dense_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    hf_id = "mattiaskvist/bge-m3-checkthat-finetuned"
    print(f"Injecting LoRA adapters from {hf_id}...")
    dense_model[0].auto_model = PeftModel.from_pretrained(
        dense_model[0].auto_model,
        hf_id,
    )
    dense_model = dense_model.to("cuda")

    print("Loading Cross-Encoder Reranker (Gemma-2B) to GPU...")
    reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-gemma")
    reranker_tokenizer.padding_side = "right"

    reranker_model = AutoModelForCausalLM.from_pretrained(
        "BAAI/bge-reranker-v2-gemma",
        dtype=torch.float16,
        device_map="auto",
    )
    reranker_model.eval()
    yes_loc = reranker_tokenizer("Yes", add_special_tokens=False)["input_ids"][0]

    # --- 2. LOAD & PREPARE COLLECTION (ONCE) ---
    print("Loading datasets...")
    collection_dataset = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        "collection",
        split="collection",
    )

    print("Preprocessing collection for indexing...")
    article_texts = []
    article_pubkeys = []
    for doc in collection_dataset:
        pubkey = doc["pubkey"]
        text_representation = article_to_text(doc)
        article_texts.append(text_representation)
        article_pubkeys.append(pubkey)

    print("Generating dense embeddings for the entire collection...")
    article_embeddings = dense_model.encode_document(
        article_texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        device="cuda",
    )

    print("Generating BM25 sparse index for the entire collection...")
    tokenized_corpus = [text.lower().split() for text in article_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # --- 3. EVALUATE ACROSS LANGUAGES ---
    languages = ["de", "fr", "en"]
    split = "dev"

    # Dictionary to hold the final big summary metrics
    global_results = {}

    # Tracker for the overall Global Average
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

        tweets = list(
            load_dataset(
                "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", lang
            )[split]
        )

        all_mrr_scores = []
        rrf_mrr_scores = []
        dense_mrr_scores = []
        sparse_mrr_scores = []

        for _, row in enumerate(
            tqdm(tweets, desc=f"Evaluating {lang.upper()} Queries")
        ):
            query_text = row["text"]
            true_pubkey = row["pubkey"]

            # Phase 1: DENSE & SPARSE RETRIEVAL
            query_embedding = dense_model.encode_query(
                query_text, convert_to_tensor=True
            )
            dense_scores = util.cos_sim(query_embedding, article_embeddings)[0]
            dense_ranks = torch.argsort(dense_scores, descending=True).tolist()

            tokenized_query = query_text.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            sparse_ranks = np.argsort(bm25_scores)[::-1].tolist()

            dense_top_5_pubkeys = [
                article_pubkeys[doc_id] for doc_id in dense_ranks[:5]
            ]
            sparse_top_5_pubkeys = [
                article_pubkeys[doc_id] for doc_id in sparse_ranks[:5]
            ]

            dense_mrr_scores.append(MRR_at_5(dense_top_5_pubkeys, true_pubkey))
            sparse_mrr_scores.append(MRR_at_5(sparse_top_5_pubkeys, true_pubkey))

            # Phase 2: RECIPROCAL RANK FUSION (RRF)
            k = 60
            rrf_scores = {}
            for rank, doc_id in enumerate(dense_ranks):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            for rank, doc_id in enumerate(sparse_ranks):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

            rrf_sorted_docs = sorted(
                rrf_scores.items(), key=lambda x: x[1], reverse=True
            )
            top_k_candidates = [doc_id for doc_id, rrf_score in rrf_sorted_docs[:10]]

            rrf_mrr_scores.append(
                MRR_at_5(
                    [article_pubkeys[doc_id] for doc_id in top_k_candidates],
                    true_pubkey,
                )
            )

            # Phase 3: CROSS-ENCODER RERANKING
            pairs = [[query_text, article_texts[doc_id]] for doc_id in top_k_candidates]
            inputs = get_inputs(pairs, reranker_tokenizer)
            inputs = inputs.to(reranker_model.device)

            with torch.inference_mode():
                outputs = reranker_model(**inputs)
                pooled_logits = last_logit_pool(
                    outputs.logits, inputs["attention_mask"]
                )
                rerank_scores = pooled_logits[:, yes_loc].cpu().float().tolist()

            final_results = list(zip(top_k_candidates, rerank_scores))
            final_results.sort(key=lambda x: x[1], reverse=True)

            predicted_pubkeys = [
                article_pubkeys[doc_id] for doc_id, score in final_results
            ]
            mrr_score = MRR_at_5(predicted_pubkeys, true_pubkey)
            all_mrr_scores.append(mrr_score)

        # Calculate averages for the current language
        avg_dense = sum(dense_mrr_scores) / len(dense_mrr_scores)
        avg_sparse = sum(sparse_mrr_scores) / len(sparse_mrr_scores)
        avg_rrf = sum(rrf_mrr_scores) / len(rrf_mrr_scores)
        avg_final = sum(all_mrr_scores) / len(all_mrr_scores)

        # Save to global results tracker
        global_results[lang] = {
            "Total Queries": len(tweets),
            "Dense MRR@5": avg_dense,
            "Sparse MRR@5": avg_sparse,
            "RRF MRR@5": avg_rrf,
            "Final MRR@5": avg_final,
        }

        # Add to global totals for the final combined average
        global_totals["queries"] += len(tweets)
        global_totals["dense_sum"] += sum(dense_mrr_scores)
        global_totals["sparse_sum"] += sum(sparse_mrr_scores)
        global_totals["rrf_sum"] += sum(rrf_mrr_scores)
        global_totals["final_sum"] += sum(all_mrr_scores)

        # Print per-language summary
        print(f"\n--- Summary for {lang.upper()} ---")
        print(f"Total Queries:      {len(tweets)}")
        print(f"Dense MRR@5:        {avg_dense:.4f}")
        print(f"Sparse MRR@5:       {avg_sparse:.4f}")
        print(f"RRF MRR@5:          {avg_rrf:.4f}")
        print(f"Final Pipeline:     {avg_final:.4f}")

    # --- 4. PRINT BIG SUMMARY ---
    print("\n\n" + "*" * 50)
    print("*" + " FINAL MULTILINGUAL EVALUATION SUMMARY ".center(48) + "*")
    print("*" * 50)

    for lang, metrics in global_results.items():
        print(f"\n[{lang.upper()}] - {metrics['Total Queries']} Queries Evaluated")
        print(f"  ├─ Dense Only:    {metrics['Dense MRR@5']:.4f}")
        print(f"  ├─ Sparse Only:   {metrics['Sparse MRR@5']:.4f}")
        print(f"  ├─ RRF Output:    {metrics['RRF MRR@5']:.4f}")
        print(f"  └─ Final Rerank:  {metrics['Final MRR@5']:.4f}")

    # Print Global Average
    total_q = global_totals["queries"]
    print("\n==================================================")
    print(f"[GLOBAL AVERAGE] - {total_q} Total Queries Across All Languages")
    print(f"  ├─ Overall Dense:    {(global_totals['dense_sum'] / total_q):.4f}")
    print(f"  ├─ Overall Sparse:   {(global_totals['sparse_sum'] / total_q):.4f}")
    print(f"  ├─ Overall RRF:      {(global_totals['rrf_sum'] / total_q):.4f}")
    print(f"  └─ Overall Final:    {(global_totals['final_sum'] / total_q):.4f}")
    print("==================================================\n")

    print("\n" + "*" * 50)


# ==========================================
# 4. ENTRY POINT
# ==========================================
@app.local_entrypoint()
def main():
    evaluate_pipeline.remote()
