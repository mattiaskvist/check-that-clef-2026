import random
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
import string


# Config
EXPERIMENT_COUNT = 2
LANG = "de"
PERCENT = 5
TOP_K = 50
K_VALUES = [3,5,25,50]

K1_VALUE = 2.0
B_VALUE = 1.0

SEEDS = list(range(1,11))
LOG_FILE = f"manual_research/research_results_{LANG}.tsv"

nltk.download("stopwords")

stemmer = LancasterStemmer()

MULTILINGUAL_STOPWORDS = set(
    stopwords.words("english") + stopwords.words("german") + stopwords.words("french")
)


# Tokenization function
def tokenize(text):
    # return text.lower().split()
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator)
    tokens = [stemmer.stem(t) for t in clean_text.split() if t not in MULTILINGUAL_STOPWORDS]
    return tokens


# Article building function
def build_article(row):
    """
    Easily change article structure here.
    """

    title = row["title"]
    abstract = row["abstract"]
    authors = row["authors"]
    venue = row["venue"]

    return title * 3 + " " + authors * 3 + " " + venue * 2 + " " + abstract


# Load datasets
def load_data():

    print("Loading datasets...")

    collection = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        "collection"
    )["collection"]

    data = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        LANG
    )

    return collection.to_list(), data["train"].to_list()


# Corpus building
def build_corpus(collection_records):

    articles = []
    pubkeys = []

    for row in collection_records:

        text = build_article(row)
        articles.append(text)
        pubkeys.append(row["pubkey"])

    tokenized = [tokenize(a) for a in articles]

    return tokenized, np.array(pubkeys)


# Query sampling for evaluation
def sample_queries(tweets, seed):

    random.seed(seed)

    sample_size = int(len(tweets) * (PERCENT / 100))

    return random.sample(tweets, sample_size)


# Evaluation
def evaluate(bm25, queries, article_pubkeys):

    results = {k:0.0 for k in K_VALUES}
    mrr_scores = []

    for tweet in tqdm(queries, leave=False):

        query = tokenize(tweet["text"])
        label = tweet["pubkey"]

        scores = bm25.get_scores(query)

        top_idx = np.argpartition(scores,-TOP_K)[-TOP_K:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        retrieved = article_pubkeys[top_idx]

        for k in K_VALUES:
            if label in retrieved[:k]:
                results[k]+=1

        top5 = retrieved[:5].tolist()

        if label in top5:
            mrr_scores.append(1/(top5.index(label)+1))
        else:
            mrr_scores.append(0)

    for k in results:
        results[k]/=len(queries)

    mrr5 = float(np.mean(mrr_scores))

    return results, mrr5


# Cross validation loop
def run_experiment(bm25, tweets, article_pubkeys):

    recall_results = {k:[] for k in K_VALUES}
    mrr_results = []

    for seed in SEEDS:

        print(f"Seed {seed}")

        queries = sample_queries(tweets, seed)

        results, mrr = evaluate(bm25, queries, article_pubkeys)

        for k,v in results.items():
            recall_results[k].append(v)

        mrr_results.append(mrr)

    return recall_results, mrr_results


# Results
def summarize(recall_results, mrr_results, experiment_count):

    # Compute mean ± std for selected metrics
    recall3_mean, recall3_std = np.mean(recall_results[3]), np.std(recall_results[3])
    recall5_mean, recall5_std = np.mean(recall_results[5]), np.std(recall_results[5])
    recall25_mean, recall25_std = np.mean(recall_results[25]), np.std(recall_results[25])
    recall50_mean, recall50_std = np.mean(recall_results[50]), np.std(recall_results[50])
    mrr5_mean, mrr5_std = np.mean(mrr_results), np.std(mrr_results)

    # Print to console
    print("\n===== FINAL RESULTS =====")
    print(f"Recall@3: {recall3_mean:.4f} ± {recall3_std:.4f}")
    print(f"Recall@5: {recall5_mean:.4f} ± {recall5_std:.4f}")
    print(f"Recall@25: {recall25_mean:.4f} ± {recall25_std:.4f}")
    print(f"Recall@50: {recall50_mean:.4f} ± {recall50_std:.4f}")
    print(f"MRR@5: {mrr5_mean:.4f} ± {mrr5_std:.4f}")

    # Save to log file (one line)
    log_line = f"{experiment_count}\t" \
               f"{recall3_mean:.4f}±{recall3_std:.4f}\t" \
               f"{recall5_mean:.4f}±{recall5_std:.4f}\t" \
               f"{recall25_mean:.4f}±{recall25_std:.4f}\t" \
               f"{recall50_mean:.4f}±{recall50_std:.4f}\t" \
               f"{mrr5_mean:.4f}±{mrr5_std:.4f}\n"

    # Append to file
    with open(LOG_FILE, "a") as f:
        f.write(log_line)


# Main
def main():

    collection_records, tweets = load_data()

    tokenized_articles, article_pubkeys = build_corpus(collection_records)

    print("Building BM25 index...")

    bm25 = BM25Okapi(
        tokenized_articles,
        k1=K1_VALUE,
        b=B_VALUE
    )

    recall_results, mrr_results = run_experiment(
        bm25,
        tweets,
        article_pubkeys
    )

    summarize(recall_results, mrr_results, experiment_count=EXPERIMENT_COUNT)


if __name__ == "__main__":
    main()