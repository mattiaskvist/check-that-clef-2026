import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from deep_translator import GoogleTranslator
from multiprocessing import Pool, cpu_count, Manager
from collections import defaultdict


# Config
EXPERIMENT_COUNT = 18
LANG = "en"
LOG_FILE = f"manual_research/research_results_{LANG}.tsv"
PERCENT = 5
TOP_K = 50
K_VALUES = [3, 5, 25, 50]

K1 = 2.25
B = 0.9

SEEDS = list(range(1, 11))

TRANSLATE_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))

stemmer = LancasterStemmer()

MULTILINGUAL_STOPWORDS = set(
    stopwords.words("english") + stopwords.words("german") + stopwords.words("french")
)

# =========================
# TOKENIZATION
# =========================


def tokenize(text):

    text = text.lower().translate(TRANSLATE_TABLE)

    tokens = [
        stemmer.stem(t)
        for t in text.split()
        if t not in MULTILINGUAL_STOPWORDS and len(t) > 1
    ]

    if len(tokens) > 1:
        tokens += [a + "_" + b for a, b in zip(tokens[:-1], tokens[1:])]

    return tokens


# Article building function
def build_article(row):
    title = row.get("title") or ""
    abstract = row.get("abstract") or ""
    venue = row.get("venue") or ""

    return " ".join([title, title, title, title, title, title, title, venue, venue, abstract])


# Load datasets
def load_data():

    print("Loading datasets...")

    collection = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "collection"
    )["collection"]

    data = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", LANG
    )

    return collection.to_list(), data["train"].to_list()


# =========================
# CORPUS BUILD
# =========================


def build_corpus(records):

    docs = []
    pubkeys = []

    for r in records:
        docs.append(tokenize(build_article(r)))
        pubkeys.append(r["pubkey"])

    return docs, np.array(pubkeys)


class FastBM25:
    def __init__(self, tokenized_docs):

        self.N = len(tokenized_docs)
        self.doc_len = np.array([len(d) for d in tokenized_docs])
        self.avgdl = self.doc_len.mean()

        self.index = defaultdict(list)
        self.df = defaultdict(int)

        for doc_id, doc in enumerate(tokenized_docs):
            freqs = defaultdict(int)
            for t in doc:
                freqs[t] += 1

            for t, tf in freqs.items():
                self.index[t].append((doc_id, tf))
                self.df[t] += 1

        self.idf = {
            t: np.log(1 + (self.N - df + 0.5) / (df + 0.5)) for t, df in self.df.items()
        }

    def get_scores(self, query):

        scores = defaultdict(float)

        for token in query:
            if token not in self.index:
                continue

            idf = self.idf[token]

            for doc_id, tf in self.index[token]:
                denom = tf + K1 * (1 - B + B * self.doc_len[doc_id] / self.avgdl)

                scores[doc_id] += idf * (tf * (K1 + 1) / denom)

        return scores



def preprocess_queries(tweets):

    queries = []

    for t in tqdm(tweets, desc="Tokenizing queries"):
        queries.append((tokenize(t["text"]), t["pubkey"]))

    return queries


def translate_tweets_if_needed(tweets):

    if LANG not in ("de", "fr"):
        return tweets

    print(f"Translating {len(tweets)} tweets from {LANG} -> en")

    translator = GoogleTranslator(source=LANG, target="en")

    for t in tqdm(tweets, desc="Translating"):
        text = t["text"]

        # deep_translator limit
        if len(text) > 5000:
            text = text[:5000]

        try:
            t["text"] = translator.translate(text)
        except Exception:
            # retry once (Google sometimes fails randomly)
            try:
                t["text"] = translator.translate(text)
            except Exception:
                t["text"] = text  # fallback: keep original

    return tweets



def evaluate_seed(args):

    seed, queries, bm25, pubkeys, counter = args

    rng = random.Random(seed)

    sample_size = int(len(queries) * (PERCENT / 100))
    sampled = rng.sample(queries, sample_size)

    results = {k: 0 for k in K_VALUES}
    mrr = []

    for tokens, label in sampled:
        scores = bm25.get_scores(tokens)

        if not scores:
            continue

        doc_ids = np.fromiter(scores.keys(), dtype=int)
        vals = np.fromiter(scores.values(), dtype=float)

        k = min(TOP_K, len(vals))

        top_idx = np.argpartition(vals, -k)[-k:]
        top_idx = top_idx[np.argsort(vals[top_idx])[::-1]]
        top = doc_ids[top_idx]

        retrieved = pubkeys[top]

        for k in K_VALUES:
            if label in retrieved[:k]:
                results[k] += 1

        top5 = retrieved[:5]
        pos = np.where(top5 == label)[0]
        mrr.append(1 / (pos[0] + 1) if len(pos) else 0)

        counter.value += 1

    for k in results:
        results[k] /= len(sampled)

    return results, np.mean(mrr)



def run_experiment(bm25, queries, pubkeys):

    manager = Manager()
    counter = manager.Value("i", 0)

    args = [(s, queries, bm25, pubkeys, counter) for s in SEEDS]

    total_queries = int(len(queries) * (PERCENT / 100)) * len(SEEDS)

    progress = tqdm(total=total_queries, desc="Evaluating queries", unit="q")

    recall_results = {k: [] for k in K_VALUES}
    mrr_results = []

    with Pool(cpu_count()) as pool:
        result_iter = pool.imap_unordered(evaluate_seed, args)

        last = 0

        while True:
            try:
                res, mrr = next(result_iter)

                for k, v in res.items():
                    recall_results[k].append(v)

                mrr_results.append(mrr)

            except StopIteration:
                break

            current = counter.value
            progress.update(current - last)
            last = current

    progress.close()

    return recall_results, mrr_results


# Results
def summarize(recall_results, mrr_results, experiment_count):

    # Compute mean ± std for selected metrics
    recall3_mean, recall3_std = np.mean(recall_results[3]), np.std(recall_results[3])
    recall5_mean, recall5_std = np.mean(recall_results[5]), np.std(recall_results[5])
    recall25_mean, recall25_std = (
        np.mean(recall_results[25]),
        np.std(recall_results[25]),
    )
    recall50_mean, recall50_std = (
        np.mean(recall_results[50]),
        np.std(recall_results[50]),
    )
    mrr5_mean, mrr5_std = np.mean(mrr_results), np.std(mrr_results)

    # Print to console
    print("\n===== FINAL RESULTS =====")
    print(f"Recall@3: {recall3_mean:.4f} ± {recall3_std:.4f}")
    print(f"Recall@5: {recall5_mean:.4f} ± {recall5_std:.4f}")
    print(f"Recall@25: {recall25_mean:.4f} ± {recall25_std:.4f}")
    print(f"Recall@50: {recall50_mean:.4f} ± {recall50_std:.4f}")
    print(f"MRR@5: {mrr5_mean:.4f} ± {mrr5_std:.4f}")

    # Save to log file (one line)
    log_line = (
        f"{experiment_count}\t"
        f"{recall3_mean:.4f}±{recall3_std:.4f}\t"
        f"{recall5_mean:.4f}±{recall5_std:.4f}\t"
        f"{recall25_mean:.4f}±{recall25_std:.4f}\t"
        f"{recall50_mean:.4f}±{recall50_std:.4f}\t"
        f"{mrr5_mean:.4f}±{mrr5_std:.4f}\n"
    )

    # Append to file
    with open(LOG_FILE, "a") as f:
        f.write(log_line)


# Main
def main():
    nltk.download("stopwords")

    collection, tweets = load_data()

    tweets = translate_tweets_if_needed(tweets)

    docs, pubkeys = build_corpus(collection)

    print("Building inverted index BM25...")
    bm25 = FastBM25(docs)

    queries = preprocess_queries(tweets)

    recall, mrr = run_experiment(bm25, queries, pubkeys)

    summarize(recall, mrr, EXPERIMENT_COUNT)


if __name__ == "__main__":
    main()
