"""Mine BM25 hard-negative triplets for dense retriever fine-tuning.

The script is intentionally executable as a one-off data-preparation command.
It loads the CheckThat train splits, mines one BM25 negative per labeled query,
and writes newline-delimited JSON triplets consumed by ``train_bge_modal.py``.
"""

import json
import sys
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# 1. Load the Collection and Query Data
print("Loading datasets...")
collection_dataset = load_dataset(
    "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "collection"
)["collection"]

english_queries = list(
    load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "en"
    )["train"]
)
german_queries = list(
    load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "de"
    )["train"]
)
french_queries = list(
    load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "fr"
    )["train"]
)

# Build a proxy map: pubkey -> English query text
# This allows DE/FR queries to "borrow" an English query for cross-lingual BM25 mining
en_proxy_map = {row["pubkey"]: row["text"] for row in english_queries}

# 2. Prepare the Collection for BM25 Indexing
print("Preparing collection...")
collection_mapping = {}
corpus_tokens = []
corpus_pubkeys = []

for doc in tqdm(
    collection_dataset,
    desc="Preparing collection for BM25",
    total=len(collection_dataset),
    unit="document",
    disable=not sys.stdout.isatty(),
):
    pubkey = doc["pubkey"]
    text_representation = f"{doc['title']}\n{doc['abstract']}"

    collection_mapping[pubkey] = text_representation
    corpus_pubkeys.append(pubkey)
    corpus_tokens.append(text_representation.lower().split())

print("Building BM25 Index...")
bm25 = BM25Okapi(corpus_tokens)

# 3. BM25 Cache Setup
# This prevents us from re-running BM25 for the DE and FR translations
hn_cache = {}


def get_cached_hard_negative(search_text, true_pubkey):
    """Runs BM25 or fetches from cache to return the hardest negative document."""
    cache_key = (search_text, true_pubkey)

    if cache_key in hn_cache:
        return hn_cache[cache_key]

    tokenized_query = search_text.lower().split()
    doc_scores = bm25.get_scores(tokenized_query)
    top_n_indices = np.argsort(doc_scores)[::-1][:10]

    for idx in top_n_indices:
        candidate_pubkey = corpus_pubkeys[idx]
        if candidate_pubkey != true_pubkey:
            negative_text = collection_mapping[candidate_pubkey]
            hn_cache[cache_key] = negative_text
            return negative_text

    return None


# 4. Mine Hard Negatives and Build Triplets
triplets = []
missing_positives = 0

print("Mining triplets for English queries...")
for row in tqdm(english_queries, desc="Processing English"):
    true_pubkey = row["pubkey"]
    if true_pubkey not in collection_mapping:
        missing_positives += 1
        continue

    positive_text = collection_mapping[true_pubkey]
    negative_text = get_cached_hard_negative(row["text"], true_pubkey)

    if negative_text:
        triplets.append(
            {
                "anchor": row["text"],
                "positive": positive_text,
                "negative": negative_text,
            }
        )


print("Mining triplets for German queries...")
for row in tqdm(german_queries, desc="Processing German"):
    true_pubkey = row["pubkey"]
    if true_pubkey not in collection_mapping:
        missing_positives += 1
        continue

    positive_text = collection_mapping[true_pubkey]

    # Borrow the English proxy text if it exists, otherwise fall back to German
    search_text = en_proxy_map.get(true_pubkey, row["text"])
    negative_text = get_cached_hard_negative(search_text, true_pubkey)

    if negative_text:
        triplets.append(
            {
                "anchor": row["text"],
                "positive": positive_text,
                "negative": negative_text,
            }
        )


print("Mining triplets for French queries...")
for row in tqdm(french_queries, desc="Processing French"):
    true_pubkey = row["pubkey"]
    if true_pubkey not in collection_mapping:
        missing_positives += 1
        continue

    positive_text = collection_mapping[true_pubkey]

    # Borrow the English proxy text if it exists, otherwise fall back to French
    search_text = en_proxy_map.get(true_pubkey, row["text"])
    negative_text = get_cached_hard_negative(search_text, true_pubkey)

    if negative_text:
        triplets.append(
            {
                "anchor": row["text"],
                "positive": positive_text,
                "negative": negative_text,
            }
        )


# 5. Save the Triplets to Disk
output_file = "hard_negative_triplets.json"
print(f"Saving {len(triplets)} triplets to {output_file}...")

with open(output_file, "w", encoding="utf-8") as f:
    for triplet in triplets:
        f.write(json.dumps(triplet, ensure_ascii=False) + "\n")

print(
    f"Done! Note: {missing_positives} queries were skipped because their target paper was not in the collection."
)
