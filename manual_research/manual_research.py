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
from collections import defaultdict, Counter


# Config
EXPERIMENT_COUNT = 24
LANG = "en"
LOG_FILE = f"manual_research/research_results_{LANG}.tsv"

PERCENT = 5
TOP_K = 50
K_VALUES = [3,5,25,50]

K1 = 2.25
B = 0.9

# diffusion
DIFFUSION_STEPS = 2
DIFFUSION_DECAY = 0.65
DIFF_NEIGHBORS = 6

# pseudo relevance feedback
PRF_DOCS = 7
PRF_TERMS = 8
PRF_WEIGHT = 0.85

SEEDS = list(range(1,11))

WINDOW_SIZE = 5

TRANSLATE_TABLE = str.maketrans(string.punctuation," "*len(string.punctuation))

stemmer = LancasterStemmer()

LANG_NLTK = "english" if LANG == "en" else ("german" if LANG == "de" else "french")

try:
    LANG_STOPWORDS = set(stopwords.words(LANG_NLTK))
except:
    nltk.download("stopwords")
    LANG_STOPWORDS = set(stopwords.words(LANG_NLTK))


# Tokenization function
def tokenize(text):

    text = text.lower().translate(TRANSLATE_TABLE)

    tokens = [
        stemmer.stem(t)
        for t in text.split()
        if t not in LANG_STOPWORDS and len(t) > 1
    ]

    if len(tokens) > 1:
        tokens += [a+"_"+b for a,b in zip(tokens[:-1],tokens[1:])]

    return tokens


# Article building function
def build_article(row):

    title = row.get("title") or ""
    abstract = row.get("abstract") or ""
    venue = row.get("venue") or ""

    return " ".join([title,title,title,venue,venue,abstract])


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


# Build corpus
def build_corpus(records):

    docs = []
    pubkeys = []

    for r in records:

        docs.append(tokenize(build_article(r)))
        pubkeys.append(r["pubkey"])

    return docs, np.array(pubkeys)

# Translate tweets if needed
def translate_tweets_if_needed(tweets):

    if LANG not in ("de","fr"):
        return tweets

    print(f"Translating {len(tweets)} tweets")

    translator = GoogleTranslator(source=LANG,target="en")

    for t in tqdm(tweets,desc="Translating"):

        text = t["text"]

        if len(text)>5000:
            text=text[:5000]

        try:
            t["text"]=translator.translate(text)
        except:
            t["text"]=text

    return tweets


# Build term graph (weighted)
def build_term_graph(docs):

    term_graph = defaultdict(Counter)

    for doc in tqdm(docs, desc="Building term graph"):

        for i,t in enumerate(doc):

            window = doc[i+1:i+WINDOW_SIZE]

            for w in window:

                if t == w:
                    continue

                term_graph[t][w] += 1
                term_graph[w][t] += 1

    return term_graph


# BM25 core
class FastBM25:
    def __init__(self,docs):

        self.docs = docs
        self.N = len(docs)

        self.doc_len = np.array([len(d) for d in docs])
        self.avgdl = self.doc_len.mean()

        self.index = defaultdict(list)
        self.df = defaultdict(int)

        for doc_id,doc in enumerate(docs):

            f = Counter(doc)

            for t,tf in f.items():
                self.index[t].append((doc_id,tf))
                self.df[t] += 1

        self.idf = {
            t: np.log(1 + (self.N - df + 0.5)/(df + 0.5))
            for t,df in self.df.items()
        }


    def get_scores(self,query):

        scores = defaultdict(float)

        for t in query:

            if t not in self.index:
                continue

            idf = self.idf[t]

            for doc_id,tf in self.index[t]:

                denom = tf + K1*(1 - B + B*self.doc_len[doc_id]/self.avgdl)

                scores[doc_id] += idf * (tf * (K1+1) / denom)

        return scores


# Diffusion (soft probability expansion)
def diffusion_expand(tokens, term_graph):

    weights = Counter({t:1.0 for t in tokens})

    for _ in range(DIFFUSION_STEPS):

        new_weights = Counter()

        for t,w in weights.items():

            neighbors = term_graph.get(t)
            if not neighbors:
                continue

            total = sum(neighbors.values()) + 1e-9

            for n,c in neighbors.most_common(DIFF_NEIGHBORS):

                new_weights[n] += w * (c / total) * DIFFUSION_DECAY

        weights.update(new_weights)

    return weights


# PRF (true weighted centroid model)
def pseudo_relevance_expand(top_docs, docs, scores):

    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)[:PRF_DOCS]

    counter = Counter()

    for doc_id,_ in ranked:

        for t in docs[doc_id]:
            counter[t] += 1

    total = sum(counter.values()) + 1e-9

    return {t: c/total for t,c in counter.most_common(PRF_TERMS)}


# Preprocess queries
def preprocess_queries(tweets):

    return [
        (tokenize(t["text"]),t["pubkey"])
        for t in tqdm(tweets,desc="Tokenizing queries")
    ]


# Rank query (fully fused model)
def rank_query(tokens, bm25, term_graph, docs):

    base_scores = bm25.get_scores(tokens)

    if not base_scores:
        return []

    diff = diffusion_expand(tokens, term_graph)

    for t,w in diff.items():

        if t not in bm25.index:
            continue

        for doc_id,_ in bm25.index[t]:
            base_scores[doc_id] += w

    doc_ids = np.fromiter(base_scores.keys(),dtype=int)
    vals = np.fromiter(base_scores.values(),dtype=float)

    k = min(TOP_K,len(vals))

    top_idx = np.argpartition(vals,-k)[-k:]
    top_idx = top_idx[np.argsort(vals[top_idx])[::-1]]

    top_docs = doc_ids[top_idx]

    prf = pseudo_relevance_expand(top_docs, docs, base_scores)

    for t,w in prf.items():

        if t not in bm25.index:
            continue

        for doc_id,_ in bm25.index[t]:
            base_scores[doc_id] += PRF_WEIGHT * w

    # final normalization (IMPORTANT)
    mx = max(base_scores.values()) + 1e-9
    for k in base_scores:
        base_scores[k] /= mx

    doc_ids = np.fromiter(base_scores.keys(),dtype=int)
    vals = np.fromiter(base_scores.values(),dtype=float)

    k = min(TOP_K,len(vals))

    top_idx = np.argpartition(vals,-k)[-k:]
    top_idx = top_idx[np.argsort(vals[top_idx])[::-1]]

    return doc_ids[top_idx]


# Evaluation
def evaluate_seed(args):

    seed,queries,bm25,pubkeys,term_graph,docs,counter = args

    rng = random.Random(seed)

    sample_size = max(1,int(len(queries)*(PERCENT/100)))
    sampled = rng.sample(queries,sample_size)

    results = {k:0 for k in K_VALUES}
    mrr = []

    for tokens,label in sampled:

        top = rank_query(tokens,bm25,term_graph,docs)
        retrieved = pubkeys[top]

        for k in K_VALUES:
            if label in retrieved[:k]:
                results[k] += 1

        pos = np.where(retrieved[:5] == label)[0]
        mrr.append(1/(pos[0]+1) if len(pos) else 0)

        counter.value += 1

    for k in results:
        results[k] /= len(sampled)

    return results,np.mean(mrr)


# Experiment runner
def run_experiment(bm25,queries,pubkeys,term_graph,docs):

    manager = Manager()
    counter = manager.Value("i",0)

    args = [
        (s,queries,bm25,pubkeys,term_graph,docs,counter)
        for s in SEEDS
    ]

    total_queries = int(len(queries)*(PERCENT/100))*len(SEEDS)

    progress = tqdm(total=total_queries,desc="Evaluating",unit="q")

    recall_results = {k:[] for k in K_VALUES}
    mrr_results = []

    with Pool(cpu_count()) as pool:

        result_iter = pool.imap_unordered(evaluate_seed,args)

        last = 0

        while True:

            try:
                res,mrr = next(result_iter)

                for k,v in res.items():
                    recall_results[k].append(v)

                mrr_results.append(mrr)

            except StopIteration:
                break

            current = counter.value
            progress.update(current-last)
            last = current

    progress.close()

    return recall_results,mrr_results


# Summary
def summarize(recall_results,mrr_results):

    recall3_mean,recall3_std = np.mean(recall_results[3]),np.std(recall_results[3])
    recall5_mean,recall5_std = np.mean(recall_results[5]),np.std(recall_results[5])
    recall25_mean,recall25_std = np.mean(recall_results[25]),np.std(recall_results[25])
    recall50_mean,recall50_std = np.mean(recall_results[50]),np.std(recall_results[50])
    mrr_mean,mrr_std = np.mean(mrr_results),np.std(mrr_results)

    print("\n===== FINAL RESULTS =====")

    print(f"Recall@3: {recall3_mean:.4f} ± {recall3_std:.4f}")
    print(f"Recall@5: {recall5_mean:.4f} ± {recall5_std:.4f}")
    print(f"Recall@25: {recall25_mean:.4f} ± {recall25_std:.4f}")
    print(f"Recall@50: {recall50_mean:.4f} ± {recall50_std:.4f}")
    print(f"MRR@5: {mrr_mean:.4f} ± {mrr_std:.4f}")

    log_line = (
        f"{EXPERIMENT_COUNT}\t"
        f"{recall3_mean:.4f}±{recall3_std:.4f}\t"
        f"{recall5_mean:.4f}±{recall5_std:.4f}\t"
        f"{recall25_mean:.4f}±{recall25_std:.4f}\t"
        f"{recall50_mean:.4f}±{recall50_std:.4f}\t"
        f"{mrr_mean:.4f}±{mrr_std:.4f}\n"
    )

    with open(LOG_FILE,"a") as f:
        f.write(log_line)


# Main
def main():

    nltk.download("stopwords")

    collection,tweets = load_data()

    tweets = translate_tweets_if_needed(tweets)

    docs,pubkeys = build_corpus(collection)

    term_graph = build_term_graph(docs)

    print("Building BM25 index...")
    bm25 = FastBM25(docs)

    queries = preprocess_queries(tweets)

    recall,mrr = run_experiment(bm25,queries,pubkeys,term_graph,docs)

    summarize(recall,mrr)


if __name__ == "__main__":
    main()
