import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, Cistem
from deep_translator import GoogleTranslator
from multiprocessing import Pool, cpu_count, Manager
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json


# Config
LANG = "en"
LOG_FILE = f"manual_research/research_results_{LANG}.tsv"

TRANSLATION_CACHE_FILE = f"translation_cache_{LANG}.json"
TRANSLATION_WORKERS = 10

TRANSLATE_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))
stemmer = SnowballStemmer("english" if LANG == "en" else "german" if LANG == "de" else "french")

# language switch
USE_ADVANCED = (LANG == "en")

if LANG == "en":
    EXPERIMENT_COUNT = 24
    PERCENT = 5
    K1 = 2.25
    B = 0.9
    DIFFUSION_STEPS = 2
    DIFFUSION_DECAY = 0.65
    DIFF_NEIGHBORS = 6
    PRF_DOCS = 7
    PRF_TERMS = 8
    PRF_WEIGHT = 0.85
else:
    EXPERIMENT_COUNT = 18
    PERCENT = 10
    K1 = 2.0
    B = 1.0

if LANG == "de":
    stemmer = Cistem()

TOP_K = 50
K_VALUES = [3, 5, 25, 50]
SEEDS = list(range(1, 11))
WINDOW_SIZE = 5


# stopwords
if LANG == "en":
    LANG_NLTK = "english"
elif LANG == "de":
    LANG_NLTK = "german"
else:
    LANG_NLTK = "french"

try:
    STOPWORDS = set(stopwords.words(LANG_NLTK))
except:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words(LANG_NLTK))


# tokenization
def tokenize(text):

    text = text.lower().translate(TRANSLATE_TABLE)

    tokens = [
        stemmer.stem(t)
        for t in text.split()
        if t not in STOPWORDS and len(t) > 1
    ]

    if len(tokens) > 1:
        tokens += [a + "_" + b for a, b in zip(tokens[:-1], tokens[1:])]

    return tokens


# article builder
def build_article(row):

    title = row.get("title") or ""
    abstract = row.get("abstract") or ""
    authors_raw = row.get("authors") or ""
    authors = (
        " ".join(str(part) for part in authors_raw)
        if isinstance(authors_raw, list)
        else str(authors_raw)
    ).strip()
    venue = row.get("venue") or ""

    if LANG == 'de':
        return " ".join([title, title, title, title, title, abstract, authors, venue, venue])

    return " ".join([title, title, title, venue, venue, abstract])


# load data
def load_data():

    collection = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        "collection"
    )["collection"]

    data = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        LANG
    )

    return collection.to_list(), data["train"].to_list()


# corpus
def build_corpus(records):

    docs = []
    pubkeys = []

    for r in records:
        docs.append(tokenize(build_article(r)))
        pubkeys.append(r["pubkey"])

    return docs, np.array(pubkeys)


# translation cache
def load_translation_cache():

    if not os.path.exists(TRANSLATION_CACHE_FILE):
        return {}

    try:
        with open(TRANSLATION_CACHE_FILE, "r") as f:
            return json.load(f)
    except:
        return {}


def save_translation_cache(cache):

    tmp = TRANSLATION_CACHE_FILE + ".tmp"

    with open(tmp, "w") as f:
        json.dump(cache, f)

    os.replace(tmp, TRANSLATION_CACHE_FILE)


# translate tweets
def translate_tweets_if_needed(tweets):

    if LANG not in ("de", "fr"):
        return tweets

    cache = load_translation_cache()

    uncached = []

    for idx, t in enumerate(tweets):

        text = t["text"][:5000]

        if text in cache:
            t["text"] = cache[text]
        else:
            uncached.append((idx, text))

    if not uncached:
        return tweets

    chunk_size = max(1, len(uncached) // TRANSLATION_WORKERS)

    chunks = [
        uncached[i:i + chunk_size]
        for i in range(0, len(uncached), chunk_size)
    ]

    def worker(chunk):

        translator = GoogleTranslator(source=LANG, target="en")

        out = []

        for idx, text in chunk:

            try:
                translated = text + translator.translate(text)
            except:
                translated = text

            out.append((idx, text, translated))

        return out

    with ThreadPoolExecutor(max_workers=TRANSLATION_WORKERS) as executor:

        futures = [
            executor.submit(worker, chunk)
            for chunk in chunks
            if len(chunk) > 0
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Translating"
        ):

            results = future.result()

            for idx, original, translated in results:

                tweets[idx]["text"] = translated
                cache[original] = translated

    save_translation_cache(cache)

    return tweets


# diffusion (only EN)
def build_term_graph(docs):

    term_graph = defaultdict(Counter)

    for doc in tqdm(docs, desc="Term graph"):

        for i, t in enumerate(doc):

            window = doc[i+1:i+WINDOW_SIZE]

            for w in window:
                if t == w:
                    continue
                term_graph[t][w] += 1
                term_graph[w][t] += 1

    return term_graph


def diffusion_expand(tokens, term_graph):

    weights = Counter({t: 1.0 for t in tokens})

    for _ in range(DIFFUSION_STEPS):

        new_weights = Counter()

        for t, w in weights.items():

            neighbors = term_graph.get(t)
            if not neighbors:
                continue

            total = sum(neighbors.values()) + 1e-9

            for n, c in neighbors.most_common(DIFF_NEIGHBORS):
                new_weights[n] += w * (c / total) * DIFFUSION_DECAY

        weights.update(new_weights)

    return weights


# BM25
class FastBM25:

    def __init__(self, docs):

        self.N = len(docs)
        self.doc_len = np.array([len(d) for d in docs])
        self.avgdl = self.doc_len.mean()

        self.index = defaultdict(list)
        self.df = defaultdict(int)

        for doc_id, doc in enumerate(docs):

            freqs = defaultdict(int)
            for t in doc:
                freqs[t] += 1

            for t, tf in freqs.items():
                self.index[t].append((doc_id, tf))
                self.df[t] += 1

        self.idf = {
            t: np.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for t, df in self.df.items()
        }

    def get_scores(self, query):

        scores = defaultdict(float)

        for t in query:

            if t not in self.index:
                continue

            idf = self.idf[t]

            for doc_id, tf in self.index[t]:

                denom = tf + K1 * (1 - B + B * self.doc_len[doc_id] / self.avgdl)

                scores[doc_id] += idf * (tf * (K1 + 1) / denom)

        return scores


# queries
def preprocess_queries(tweets):

    return [
        (tokenize(t["text"]), t["pubkey"])
        for t in tqdm(tweets, desc="Queries")
    ]


# ranking
def rank_query(tokens, bm25, term_graph, docs):

    scores = bm25.get_scores(tokens)

    if not scores:
        return []

    # diffusion only EN
    if USE_ADVANCED:

        diff = diffusion_expand(tokens, term_graph)

        for t, w in diff.items():
            if t not in bm25.index:
                continue
            for doc_id, _ in bm25.index[t]:
                scores[doc_id] += w

        doc_ids = np.fromiter(scores.keys(), dtype=int)
        vals = np.fromiter(scores.values(), dtype=float)

        k = min(TOP_K, len(vals))
        top_idx = np.argpartition(vals, -k)[-k:]
        top_idx = top_idx[np.argsort(vals[top_idx])[::-1]]

        top_docs = doc_ids[top_idx]

        # PRF
        counter = Counter()

        for doc_id in top_docs:
            for t in docs[doc_id]:
                counter[t] += 1

        total = sum(counter.values()) + 1e-9

        for t, c in counter.most_common(8):
            if t not in bm25.index:
                continue
            w = (c / total)
            for doc_id, _ in bm25.index[t]:
                scores[doc_id] += PRF_WEIGHT * w

    # normalize
    mx = max(scores.values()) + 1e-9
    for k in scores:
        scores[k] /= mx

    doc_ids = np.fromiter(scores.keys(), dtype=int)
    vals = np.fromiter(scores.values(), dtype=float)

    k = min(TOP_K, len(vals))
    top_idx = np.argpartition(vals, -k)[-k:]
    top_idx = top_idx[np.argsort(vals[top_idx])[::-1]]

    return doc_ids[top_idx]


# evaluation
def evaluate_seed(args):

    seed, queries, bm25, pubkeys, term_graph, docs, counter = args

    rng = random.Random(seed)

    sample_size = max(1, int(len(queries) * (PERCENT / 100)))
    sampled = rng.sample(queries, sample_size)

    results = {k: 0 for k in K_VALUES}
    mrr = []

    for tokens, label in sampled:

        top = rank_query(tokens, bm25, term_graph, docs)
        retrieved = pubkeys[top]

        for k in K_VALUES:
            if label in retrieved[:k]:
                results[k] += 1

        pos = np.where(retrieved[:5] == label)[0]
        mrr.append(1 / (pos[0] + 1) if len(pos) else 0)

        counter.value += 1

    for k in results:
        results[k] /= len(sampled)

    return results, np.mean(mrr)


# runner
def run_experiment(bm25, queries, pubkeys, term_graph, docs):

    manager = Manager()
    counter = manager.Value("i", 0)

    args = [
        (s, queries, bm25, pubkeys, term_graph, docs, counter)
        for s in SEEDS
    ]

    total = int(len(queries) * (PERCENT / 100)) * len(SEEDS)

    progress = tqdm(total=total, desc="Eval", unit="q")

    recall_results = {k: [] for k in K_VALUES}
    mrr_results = []

    with Pool(cpu_count()) as pool:

        for res, mrr in pool.imap_unordered(evaluate_seed, args):

            for k, v in res.items():
                recall_results[k].append(v)

            mrr_results.append(mrr)

    progress.close()

    return recall_results, mrr_results


# summary
def summarize(recall_results, mrr_results):

    print("\n===== FINAL RESULTS =====")

    for k in K_VALUES:
        print(f"Recall@{k}: {np.mean(recall_results[k]):.4f} ± {np.std(recall_results[k]):.4f}")

    print(f"MRR@5: {np.mean(mrr_results):.4f} ± {np.std(mrr_results):.4f}")

    log_line = (
        f"{EXPERIMENT_COUNT}\t"
        f"{np.mean(recall_results[3]):.4f}\t"
        f"{np.mean(recall_results[5]):.4f}\t"
        f"{np.mean(recall_results[25]):.4f}\t"
        f"{np.mean(recall_results[50]):.4f}\t"
        f"{np.mean(mrr_results):.4f}\n"
    )

    with open(LOG_FILE, "a") as f:
        f.write(log_line)


# main
def main():

    nltk.download("stopwords")

    collection, tweets = load_data()

    tweets = translate_tweets_if_needed(tweets)

    docs, pubkeys = build_corpus(collection)

    term_graph = build_term_graph(docs) if USE_ADVANCED else None

    bm25 = FastBM25(docs)

    queries = preprocess_queries(tweets)

    recall, mrr = run_experiment(bm25, queries, pubkeys, term_graph, docs)

    summarize(recall, mrr)


if __name__ == "__main__":
    main()