from datasets import load_dataset
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import tqdm
from scorer import scorer

LANG = "en"
TOP_K = 5

collection_data = load_dataset(
    "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
    "collection"
)["collection"]

data = load_dataset(
    "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
    LANG
)

all_tweets = data["train"].to_list()  # convert to list to avoid typing issues
collection_records = collection_data.to_list()

articles = [(row["title"] + " ") * 3 + row["abstract"] for row in collection_records]
article_pubkeys = np.array([row["pubkey"] for row in collection_records])

def tokenize(text: str):
    return text.lower().split()

tokenized_articles = [tokenize(a) for a in articles]

bm25 = BM25Okapi(tokenized_articles, k1=2.0, b=1.0)

top5_preds = []

n = len(all_tweets)
with tqdm(total=n, mininterval=n//100 or 1) as pbar:  # log roughly once per 1%
    for tweet in all_tweets:
        tweet = dict(tweet)  # cast to dict to satisfy Pylance
        query = tokenize(tweet.get("text", ""))
        scores = bm25.get_scores(query)
        top_indices = np.argpartition(scores, -TOP_K)[-TOP_K:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        top5_preds.append(list(article_pubkeys[top_indices[:5]]))
        pbar.update(1)

mrr5_score = scorer(top5_preds, LANG, "train")
print(f"MRR@5 on entire dataset: {mrr5_score:.4f}")