import random
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

LANG = "en"
PERCENT = 5
TOP_K = 25
K_VALUES = [3, 5, 10, 15, 25]

K1_VALUES = [2.0]
B_VALUES = [1.0]

collection_data = load_dataset(
    "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
    "collection"
)["collection"]

data = load_dataset(
    "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
    LANG
)

dev_split = data["train"]

collection_records = collection_data.to_list()
articles = [(row["title"] + " ") * 3 + row["abstract"] for row in collection_records]
article_pubkeys = np.array([row["pubkey"] for row in collection_records])

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def tokenize(text, improved=False):
    tokens = text.lower().split()
    if improved:
        tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return tokens

# Define models to test
models = {
    "baseline": {
        "tokenized_articles": [tokenize(a, improved=False) for a in articles],
        "improved": False
    },
    "stopword_stem": {
        "tokenized_articles": [tokenize(a, improved=True) for a in articles],
        "improved": True
    },
    "title_weighted": {
        "tokenized_articles": [
            tokenize((row["title"] + " ") * 5 + row["abstract"], improved=True)
            for row in collection_records
        ],
        "improved": True
    },
    "no_stopwords_only": {
        "tokenized_articles": [
            [t for t in tokenize(a, improved=False) if t not in stop_words]
            for a in articles
        ],
        "improved": False
    }
}

tweet_data = dev_split.to_list()
sample_size = int(len(tweet_data) * (PERCENT / 100))
random.seed(13)
tweet_sample = random.sample(tweet_data, sample_size)
labels = [tweet["pubkey"] for tweet in tweet_sample]

def evaluate_bm25(bm25, tweets, improved=False):
    results = {k: 0.0 for k in K_VALUES}
    top5_predictions = []

    n = len(tweets)
    with tqdm(total=n, mininterval=n//100 or 1) as pbar:
        for tweet in tweets:
            query = tokenize(tweet["text"], improved=improved)
            true_pubkey = tweet["pubkey"]
            scores = bm25.get_scores(query)
            top_indices = np.argpartition(scores, -TOP_K)[-TOP_K:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            retrieved_pubkeys = article_pubkeys[top_indices]
            top5_predictions.append(list(retrieved_pubkeys[:5]))
            for k in K_VALUES:
                if true_pubkey in retrieved_pubkeys[:k]:
                    results[k] += 1
            pbar.update(1)

    for k in results:
        results[k] /= n
    return results, top5_predictions

def compute_mrr5(top5_preds, labels):
    scores = []
    for preds, label in zip(top5_preds, labels):
        if label in preds:
            rank = preds.index(label) + 1
            scores.append(1 / rank)
        else:
            scores.append(0)
    return sum(scores) / len(scores)

for k1 in K1_VALUES:
    for b in B_VALUES:
        print(f"\nTesting k1={k1}, b={b} for all models...\n")
        comparison_results = {}
        for model_name, model_data in models.items():
            bm25 = BM25Okapi(model_data["tokenized_articles"], k1=k1, b=b)
            scores, top5_preds = evaluate_bm25(bm25, tweet_sample, improved=model_data["improved"])
            mrr5 = compute_mrr5(top5_preds, labels)
            comparison_results[model_name] = (mrr5, scores)

        # Print comparison table
        print(f"{'Model':20} | {'MRR@5':7} | " + " | ".join([f"R@{k}" for k in K_VALUES]))
        print("-" * (22 + 8 + len(K_VALUES) * 8))
        for model_name, (mrr, scores) in comparison_results.items():
            recall_str = " | ".join([f"{scores[k]:.4f}" for k in K_VALUES])
            print(f"{model_name:20} | {mrr:.4f} | {recall_str}")
        print("-" * (22 + 8 + len(K_VALUES) * 8))