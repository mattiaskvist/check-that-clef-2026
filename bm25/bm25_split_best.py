import random
import re
from collections import Counter, defaultdict
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer

# ------------------------------------------------------------
# Goal: maximize recall@25 using many complementary methods.
# This script builds multiple BM25 indices + query variants,
# uses pseudo-relevance feedback, and fuses scores.
# ------------------------------------------------------------

LANG = "en"
PERCENT = 1
TOP_K = 25
K_VALUES = [3, 5, 10, 15, 25]

K1 = 2.0
B = 1.0

CANDIDATE_K = 10000  # per-model top-k to pool (high recall mode)
RRF_K = 60         # RRF smoothing
PRF_TOP_DOCS = 30
PRF_TERMS = 30
USE_THREADS = True
NUM_WORKERS = min(8, cpu_count())
USE_SYNONYMS = False  # WordNet is slow and not thread-safe here

random.seed(13)
np.random.seed(13)

# --- dataset ---
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

article_pubkeys = np.array([row["pubkey"] for row in collection_records])

# Precompute fields
raw_titles = [row["title"] or "" for row in collection_records]
raw_abstracts = [row["abstract"] or "" for row in collection_records]
raw_concat = [t + " " + a for t, a in zip(raw_titles, raw_abstracts)]

# --- text utils ---
stemmer = PorterStemmer()

try:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set()

# WordNet is optional (may not exist offline)
try:
    nltk.download("wordnet", quiet=True)
    _WORDNET_OK = True
except Exception:
    _WORDNET_OK = False

WORD_RE = re.compile(r"[a-z0-9]+")
EMPTY_TOKEN = "__empty__"


def normalize_text(text: str) -> str:
    return text.lower()


def word_tokens(text: str) -> List[str]:
    return WORD_RE.findall(normalize_text(text))


def remove_stopwords(tokens: List[str]) -> List[str]:
    if not STOP_WORDS:
        return tokens
    return [t for t in tokens if t not in STOP_WORDS]


def stem_tokens(tokens: List[str]) -> List[str]:
    return [stemmer.stem(t) for t in tokens]


def make_bigrams(tokens: List[str]) -> List[str]:
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


def char_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> List[str]:
    t = re.sub(r"\s+", " ", normalize_text(text)).strip()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    t = t.replace(" ", "_")
    grams = []
    for n in range(min_n, max_n + 1):
        if len(t) < n:
            continue
        grams.extend([t[i:i+n] for i in range(len(t) - n + 1)])
    return grams


def expand_with_synonyms(tokens: List[str], max_per_token: int = 2) -> List[str]:
    if not USE_SYNONYMS or not _WORDNET_OK:
        return []
    expanded = []
    seen = set()
    for t in tokens:
        if len(t) < 4 or t in seen:
            continue
        seen.add(t)
        syns = set()
        try:
            for syn in wordnet.synsets(t):
                if syn is None:
                    continue
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ").lower()
                    if name != t:
                        syns.add(name)
                    if len(syns) >= max_per_token:
                        break
                if len(syns) >= max_per_token:
                    break
        except Exception:
            # WordNet can be flaky under threading; skip on any error.
            continue
        for s in syns:
            expanded.extend(word_tokens(s))
    return expanded


def ensure_non_empty(tokens: List[str]) -> List[str]:
    return tokens if tokens else [EMPTY_TOKEN]


# --- tokenization variants for documents ---

def build_doc_variants(title: str, abstract: str) -> Dict[str, List[str]]:
    base_text = f"{title} {abstract}"
    base_tokens = word_tokens(base_text)
    no_stop = remove_stopwords(base_tokens)
    stem = stem_tokens(base_tokens)
    stem_no_stop = stem_tokens(no_stop)

    title_tokens = word_tokens(title)
    title_weighted = title_tokens * 5 + base_tokens
    title_weighted = remove_stopwords(title_weighted)

    bigrams = make_bigrams(base_tokens)
    bigrams_ns = make_bigrams(no_stop)

    chargrams = char_ngrams(base_text)

    return {
        "baseline": ensure_non_empty(base_tokens),
        "no_stop": ensure_non_empty(no_stop),
        "stem": ensure_non_empty(stem),
        "stem_no_stop": ensure_non_empty(stem_no_stop),
        "title_weighted": ensure_non_empty(title_weighted),
        "title_only": ensure_non_empty(title_tokens),
        "abstract_only": ensure_non_empty(word_tokens(abstract)),
        "bigrams": ensure_non_empty(bigrams),
        "bigrams_ns": ensure_non_empty(bigrams_ns),
        "chargrams": ensure_non_empty(chargrams),
    }


# Precompute doc variants
all_doc_variants = [build_doc_variants(t, a) for t, a in zip(raw_titles, raw_abstracts)]

# --- build BM25 models ---
MODEL_SPECS = [
    "baseline",
    "stem",
    "stem_no_stop",
    "title_weighted",
    "title_only",
]

bm25_models: Dict[str, BM25Okapi] = {}
bm25_corpora: Dict[str, List[List[str]]] = {}

for name in MODEL_SPECS:
    tokenized_articles = [doc[name] for doc in all_doc_variants]
    bm25_models[name] = BM25Okapi(tokenized_articles, k1=K1, b=B)
    bm25_corpora[name] = tokenized_articles


# --- query variants ---

def build_query_variants(text: str) -> Dict[str, List[str]]:
    base_tokens = word_tokens(text)
    no_stop = remove_stopwords(base_tokens)
    stem = stem_tokens(base_tokens)
    stem_no_stop = stem_tokens(no_stop)
    bigrams = make_bigrams(base_tokens)
    bigrams_ns = make_bigrams(no_stop)
    chargrams = char_ngrams(text)
    synonyms = expand_with_synonyms(no_stop)

    return {
        "baseline": base_tokens,
        "no_stop": no_stop,
        "stem": stem,
        "stem_no_stop": stem_no_stop,
        "bigrams": bigrams,
        "bigrams_ns": bigrams_ns,
        "chargrams": chargrams,
        "synonyms": synonyms,
    }


def pseudo_relevance_feedback(
    bm25: BM25Okapi,
    corpus: List[List[str]],
    base_query: List[str],
    top_docs: int = PRF_TOP_DOCS,
    n_terms: int = PRF_TERMS,
) -> List[str]:
    if not base_query:
        return []
    scores = bm25.get_scores(base_query)
    top_indices = np.argpartition(scores, -top_docs)[-top_docs:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    term_scores = Counter()
    for idx in top_indices:
        tokens = corpus[idx]
        tf = Counter(tokens)
        for t, c in tf.items():
            if t in STOP_WORDS or len(t) < 3:
                continue
            term_scores[t] += c * bm25.idf.get(t, 0.0)

    expanded = [t for t, _ in term_scores.most_common(n_terms)]
    return expanded


# --- evaluation ---

tweet_data = dev_split.to_list()
sample_size = int(len(tweet_data) * (PERCENT / 100))
tweet_sample = random.sample(tweet_data, sample_size)
labels = [tweet["pubkey"] for tweet in tweet_sample]


def fuse_scores(score_lists: Dict[str, Dict[int, float]]) -> Dict[int, float]:
    # score_lists: model -> {doc_idx: score}
    fused = defaultdict(float)

    # min-max normalize per model
    for model, scores in score_lists.items():
        if not scores:
            continue
        vals = list(scores.values())
        vmin = min(vals)
        vmax = max(vals)
        denom = (vmax - vmin) if vmax > vmin else 1.0
        for doc_idx, s in scores.items():
            norm = (s - vmin) / denom
            fused[doc_idx] += norm

    return fused


def rrf_fusion(rank_lists: Dict[str, List[int]], k: int = RRF_K) -> Dict[int, float]:
    fused = defaultdict(float)
    for _, ranked in rank_lists.items():
        for r, doc_idx in enumerate(ranked):
            fused[doc_idx] += 1.0 / (k + r + 1)
    return fused


def evaluate_bm25_split(tweets):
    results = {k: 0.0 for k in K_VALUES}
    top5_predictions = []

    def process_tweet(tweet):
        text = tweet["text"]
        true_pubkey = tweet["pubkey"]

        qv = build_query_variants(text)

        # PRF expansion on baseline
        prf_terms = pseudo_relevance_feedback(
            bm25_models["baseline"],
            bm25_corpora["baseline"],
            qv["baseline"],
            top_docs=PRF_TOP_DOCS,
            n_terms=PRF_TERMS,
        )
        expanded_query = qv["baseline"] + prf_terms + qv.get("synonyms", [])

        # Collect scores from multiple models
        score_lists: Dict[str, Dict[int, float]] = {}
        rank_lists: Dict[str, List[int]] = {}

        model_queries = {
            "baseline": qv["baseline"],
            "stem": qv["stem"],
            "stem_no_stop": qv["stem_no_stop"],
            "title_weighted": qv["stem_no_stop"],
            "title_only": qv["no_stop"],
        }

        for model_name, query in model_queries.items():
            if not query:
                continue
            bm25 = bm25_models[model_name]
            scores = bm25.get_scores(query)
            top_indices = np.argpartition(scores, -CANDIDATE_K)[-CANDIDATE_K:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            score_lists[model_name] = {int(i): float(scores[i]) for i in top_indices}
            rank_lists[model_name] = list(top_indices)

        # Add PRF-expanded baseline as another "model"
        if expanded_query:
            scores = bm25_models["baseline"].get_scores(expanded_query)
            top_indices = np.argpartition(scores, -CANDIDATE_K)[-CANDIDATE_K:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            score_lists["baseline_prf"] = {int(i): float(scores[i]) for i in top_indices}
            rank_lists["baseline_prf"] = list(top_indices)

        # Score fusion + RRF fusion
        fused_scores = fuse_scores(score_lists)
        rrf_scores = rrf_fusion(rank_lists)

        # Combine
        for doc_idx, s in rrf_scores.items():
            fused_scores[doc_idx] += s

        # Candidate set
        if fused_scores:
            ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            top_indices = [i for i, _ in ranked[:TOP_K]]
        else:
            top_indices = []

        retrieved_pubkeys = article_pubkeys[top_indices]
        top5 = list(retrieved_pubkeys[:5])
        hits = {k: (1 if true_pubkey in retrieved_pubkeys[:k] else 0) for k in K_VALUES}
        return hits, top5

    n = len(tweets)
    with tqdm(total=n, mininterval=n//100 or 1) as pbar:
        if USE_THREADS and NUM_WORKERS > 1:
            with ThreadPool(NUM_WORKERS) as pool:
                for hits, top5 in pool.imap_unordered(process_tweet, tweets):
                    for k in K_VALUES:
                        results[k] += hits[k]
                    top5_predictions.append(top5)
                    pbar.update(1)
        else:
            for tweet in tweets:
                hits, top5 = process_tweet(tweet)
                for k in K_VALUES:
                    results[k] += hits[k]
                top5_predictions.append(top5)
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


if __name__ == "__main__":
    print("Running bm25_split_best with aggressive recall maximization...")
    scores, top5_preds = evaluate_bm25_split(tweet_sample)
    mrr5 = compute_mrr5(top5_preds, labels)

    print(f"{'Model':20} | {'MRR@5':7} | " + " | ".join([f"R@{k}" for k in K_VALUES]))
    print("-" * (22 + 8 + len(K_VALUES) * 8))
    recall_str = " | ".join([f"{scores[k]:.4f}" for k in K_VALUES])
    print(f"{'split_best_fused':20} | {mrr5:.4f} | {recall_str}")
    print("-" * (22 + 8 + len(K_VALUES) * 8))
