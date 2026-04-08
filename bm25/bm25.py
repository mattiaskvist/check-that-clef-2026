import random
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm


# Config 
LANG = "en"
PERCENT = 10          # percent of tweets to test
TOP_K = 25

K_VALUES = [3, 5, 10, 15, 25]

# hyperparameter search space
K1_VALUES = [0.8, 1.2, 1.6, 2.0]
B_VALUES = [0.4, 0.6, 0.75, 0.9]

# Load datasets
collection_data = load_dataset(
    "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
    "collection"
)["collection"]

data = load_dataset(
    "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
    LANG
)

dev_split = data["dev"]


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
tweet_sample = random.sample(tweet_data, sample_size)


# Evaluation function
def evaluate_bm25(bm25):

    results = {k: 0.0 for k in K_VALUES}

    for tweet in tqdm(tweet_sample):

        query = tokenize(tweet["text"])
        true_pubkey = tweet["pubkey"]

        scores = bm25.get_scores(query)
        top_indices = np.argpartition(scores, -TOP_K)[-TOP_K:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        retrieved_pubkeys = article_pubkeys[top_indices]

        for k in K_VALUES:
            if true_pubkey in retrieved_pubkeys[:k]:
                results[k] += 1

    for k in results:
        results[k] /= len(tweet_sample)

    return results


# BM25 Baseline
print("Running baseline BM25...")

bm25 = BM25Okapi(tokenized_articles)

# baseline_results = evaluate_bm25(bm25)

# print("\nBaseline Recall:")
# for k, v in baseline_results.items():
#     print(f"Recall@{k}: {v:.4f}")


# Hyperparameter search
print("\nHyperparameter search...")

grid_results = []

for k1 in K1_VALUES:
    for b in B_VALUES:

        print(f"Testing k1={k1}, b={b}")

        bm25 = BM25Okapi(
            tokenized_articles,
            k1=k1,
            b=b
        )

        scores = evaluate_bm25(bm25)

        # grid_results.append({
        #     "k1": k1,
        #     "b": b,
        #     **scores
        # })

        print(f"Results for k1={k1}, b={b}:")
        for k, v in scores.items():
            print(f"Recall@{k}: {v:.4f}")
        print("-" * 30)




# # Graphing
# def plot_metric(metric):

#     xs = []
#     ys = []

#     for r in grid_results:
#         xs.append(r["k1"] + r["b"])
#         ys.append(r[metric])

#     plt.scatter(xs, ys)

#     plt.xlabel("k1 + b (parameter variation)")
#     plt.ylabel(metric)
#     plt.title(f"BM25 Parameter Impact on {metric}")

#     plt.show()


# for k in K_VALUES:
#     plot_metric(f"{k}")