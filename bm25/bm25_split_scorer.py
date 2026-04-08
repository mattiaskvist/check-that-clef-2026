import random
import numpy as np

from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm


# Config 
LANG = "en"
PERCENT = 10          # percent of tweets to test
TOP_K = 25

K_VALUES = [3, 5, 10, 15, 25]

# hyperparameter search space
K1_VALUES = [2.0]
B_VALUES = [1.0]

# Load datasets
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

articles = []
article_pubkeys = []

for row in collection_records:
    text = (row["title"] + " ") * 3 + row["abstract"]
    articles.append(text)
    article_pubkeys.append(row["pubkey"])

article_pubkeys = np.array(article_pubkeys)


# Tokenization function
def tokenize(text):
    return text.lower().split()


tokenized_articles = [tokenize(a) for a in articles]


# Sample tweets for evaluation
tweet_data = dev_split.to_list()

sample_size = int(len(tweet_data) * (PERCENT / 100))
random.seed(13)
tweet_sample = random.sample(tweet_data, sample_size)


# Evaluation function
def evaluate_bm25(bm25):

    results = {k: 0.0 for k in K_VALUES}
    top5_predictions = []

    for tweet in tqdm(tweet_sample):

        query = tokenize(tweet["text"])
        true_pubkey = tweet["pubkey"]

        scores = bm25.get_scores(query)

        top_indices = np.argpartition(scores, -TOP_K)[-TOP_K:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        retrieved_pubkeys = article_pubkeys[top_indices]

        # store top5 for MRR
        top5_predictions.append(list(retrieved_pubkeys[:5]))

        for k in K_VALUES:
            if true_pubkey in retrieved_pubkeys[:k]:
                results[k] += 1

    for k in results:
        results[k] /= len(tweet_sample)

    return results, top5_predictions

# Hyperparameter search
print("\nHyperparameter search...")

def compute_mrr5(top5_preds, labels):

    scores = []

    for preds, label in zip(top5_preds, labels):

        if label in preds:
            rank = preds.index(label) + 1
            scores.append(1 / rank)
        else:
            scores.append(0)

    return sum(scores) / len(scores)

grid_results = []

for k1 in K1_VALUES:
    for b in B_VALUES:

        print(f"Testing k1={k1}, b={b}")

        bm25 = BM25Okapi(
            tokenized_articles,
            k1=k1,
            b=b
        )

        scores, top5_preds = evaluate_bm25(bm25)

        labels = [tweet["pubkey"] for tweet in tweet_sample]

        mrr5 = compute_mrr5(top5_preds, labels)
        print(f"Results for k1={k1}, b={b}:")
        print(f"MRR@5: {mrr5:.4f}")

        for k, v in scores.items():
            print(f"Recall@{k}: {v:.4f}")
        print("-" * 30)

