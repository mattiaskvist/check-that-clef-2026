import numpy as np
from datasets import load_dataset, disable_progress_bar
from rank_bm25 import BM25Plus
from tqdm import tqdm
import re
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords

from full_pipeline.interfaces import BaseRetriever

# Use NLTK stopwords for multiple languages
STOPWORDS = frozenset(
    stopwords.words("english") + stopwords.words("german") + stopwords.words("french")
)

# Disable Hugging Face progress bars so they don't pollute the agent's run.log
disable_progress_bar()

CHECKTHAT_DATASET = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"
SAMPLE_SIZE = 500  # Stratified sample size per language


class SparseRetriever(BaseRetriever):
    """A retriever that performs sparse retrieval."""

    def __init__(self):
        self.bm25_model = None
        self.stemmer = LancasterStemmer()

    def tokenize(self, text: str, add_bigrams: bool = True) -> list[str]:
        """Tokenize text with punctuation, stopword removal, stemming, and optional bigrams."""
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = text.split()
        unigrams = [self.stemmer.stem(t) for t in tokens if t not in STOPWORDS]
        if add_bigrams and len(unigrams) >= 2:
            bigrams = [
                f"{unigrams[i]}_{unigrams[i + 1]}" for i in range(len(unigrams) - 1)
            ]
            return unigrams + bigrams
        return unigrams

    def index(self, collection: list[dict]):
        corpus = [self.document_to_text(doc) for doc in collection]
        tokenized_corpus = [self.tokenize(text) for text in corpus]
        self.bm25_model = BM25Plus(tokenized_corpus, k1=2.5, b=0.85)

    def search(self, query: str) -> list[int]:
        tokenized_query = self.tokenize(query)
        scores = self.bm25_model.get_scores(tokenized_query)
        return np.argsort(scores)[::-1].tolist()

    def document_to_text(self, doc: dict) -> str:
        """Turn the article dict into a single string for indexing and retrieval.

        Args:
            doc (dict): dict with keys "title", "abstract", "pubkey", "authors", "venue".
                Authors is may be just names, but could also include universities, addresses, etc.
                Venue is the name of the conference/journal where the article was published. Both may be empty strings.

        Returns:
            str: A single string representation of the article
        """
        title = (doc.get("title") or "").strip()
        abstract = (doc.get("abstract") or "").strip()
        authors_raw = doc.get("authors") or ""
        authors = (
            " ".join(str(part) for part in authors_raw)
            if isinstance(authors_raw, list)
            else str(authors_raw)
        ).strip()
        venue = str(doc.get("venue") or "").strip()
        # Repeat title to boost its importance
        return f"{title} {title} {title} {abstract} {authors} {venue}".strip()


def top_k_pubkeys_for_queries(
    model, queries: list[dict], article_pubkeys: list[str], k: int = 30
) -> list[list[str]]:
    top_k_preds: list[list[str]] = []

    # tqdm disabled to prevent context-window bloat in run.log
    for row in tqdm(queries, desc="Predicting", disable=True):
        top_k_doc_indices = model.search(row["text"])[:k]
        top_k_preds.append([article_pubkeys[idx] for idx in top_k_doc_indices])

    return top_k_preds


def recall_at_k(preds: list[list[str]], targets: list[str], k: int = 30) -> float:
    """
    Calculates Recall@K (Hit Rate) for single-target queries.
    Returns the percentage of queries where the target pubkey was in the top K predictions.
    """
    hits = 0
    for pred_list, target in zip(preds, targets):
        if target in pred_list[:k]:
            hits += 1

    return hits / len(targets) if targets else 0.0


def main() -> None:
    split = "train"
    collection = load_dataset(CHECKTHAT_DATASET, "collection", split="collection")

    article_pubkeys = [doc["pubkey"] for doc in collection]
    model = SparseRetriever()

    model.index(collection)

    all_results: list[dict] = []

    # German is the worst performing language, french is second and english is best.
    # This is most likely because the documents are mainly in english, so the sparse
    # retriever has more lexical overlap to latch onto for english queries.
    for lang in ["de", "fr", "en"]:
        # Load full language dataset
        hf_dataset = load_dataset(CHECKTHAT_DATASET, lang, split=split)
        
        # Safely cap at SAMPLE_SIZE in case a train set is slightly under 500
        safe_sample_size = min(SAMPLE_SIZE, len(hf_dataset))
        
        # Shuffle with a fixed seed and select the sample
        sampled_dataset = hf_dataset.shuffle(seed=42).select(range(safe_sample_size))
        
        queries = list(sampled_dataset)
        targets = [q["pubkey"] for q in queries]
        num_queries = len(queries)

        top30_preds = top_k_pubkeys_for_queries(model, queries, article_pubkeys, k=30)
        score = recall_at_k(preds=top30_preds, targets=targets, k=30)

        all_results.append(
            {"lang": lang, "recall30": score, "num_queries": num_queries}
        )

        # Keep a minimal log for humans reading the run.log
        print(f"{lang.upper()} Recall@30: {score:.6f} (Subset Queries: {num_queries})")

    # Calculate the query-weighted average 
    total_queries = sum(r["num_queries"] for r in all_results)
    total_recall_sum = sum(r["recall30"] * r["num_queries"] for r in all_results)

    balanced_avg_recall30 = total_recall_sum / total_queries if total_queries > 0 else 0.0

    # The distinct target line for the autoresearch agent
    print(f"RECALL@30: {balanced_avg_recall30:.6f}")


if __name__ == "__main__":
    main()
