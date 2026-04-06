import numpy as np
from datasets import load_dataset, disable_progress_bar
from rank_bm25 import BM25Plus
from tqdm import tqdm
import re
from nltk.stem.snowball import SnowballStemmer

from full_pipeline.interfaces import BaseRetriever
from scorer import scorer

# Initialize stemmers for each language
STEMMERS = {
    "en": SnowballStemmer("english"),
    "de": SnowballStemmer("german"),
    "fr": SnowballStemmer("french"),
}

# Simple multilingual stopwords (high frequency, low information)
STOPWORDS = frozenset([
    # English - expanded
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "of", "in", "to",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "and", "or", "but", "if", "then", "than", "that", "this", "it",
    "we", "they", "he", "she", "you", "i", "our", "their", "its", "my",
    "your", "his", "her", "who", "which", "what", "when", "where", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "also", "just",
    "about", "after", "before", "between", "during", "without", "within",
    "there", "here", "these", "those", "very", "too", "also", "well", "back",
    # German - expanded
    "der", "die", "das", "ein", "eine", "ist", "sind", "war", "waren",
    "und", "oder", "aber", "wenn", "dann", "von", "mit", "auf", "für",
    "zu", "bei", "nach", "aus", "im", "am", "um", "als", "wie", "so",
    "wir", "sie", "er", "es", "ich", "du", "ihr", "uns", "euch", "sich",
    "sein", "haben", "werden", "kann", "muss", "soll", "will", "darf",
    "nicht", "auch", "nur", "noch", "schon", "mehr", "sehr", "hier", "dort",
    # French - expanded
    "le", "la", "les", "un", "une", "est", "sont", "était", "étaient",
    "et", "ou", "mais", "si", "alors", "de", "du", "des", "en", "à",
    "pour", "par", "sur", "dans", "avec", "ce", "cette", "qui", "que",
    "nous", "vous", "ils", "elles", "il", "elle", "je", "tu", "on",
    "son", "sa", "ses", "notre", "votre", "leur", "leurs", "mon", "ma",
    "ne", "pas", "plus", "moins", "très", "bien", "aussi", "encore", "tout",
])

# Disable Hugging Face progress bars so they don't pollute the agent's run.log
disable_progress_bar()

CHECKTHAT_DATASET = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"


class SparseRetriever(BaseRetriever):
    """A retriever that performs sparse retrieval.

    Args:
        BaseRetriever: The base retriever interface that this class implements.
    """

    def __init__(self):
        self.bm25_model = None
        self.stemmer = STEMMERS["en"]  # Corpus is in English

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text with punctuation, stopword removal, and stemming."""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [self.stemmer.stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]

    def index(self, collection: list[dict]):
        corpus = [self.document_to_text(doc) for doc in collection]
        tokenized_corpus = [self.tokenize(text) for text in corpus]
        self.bm25_model = BM25Plus(tokenized_corpus, k1=2.5, b=0.9)

    def search(self, query: str) -> list[int]:
        tokenized_query = self.tokenize(query)
        scores = self.bm25_model.get_scores(tokenized_query)
        return np.argsort(scores)[::-1].tolist()

    def document_to_text(self, doc: dict) -> str:
        title = (doc.get("title") or "").strip()
        abstract = (doc.get("abstract") or "").strip()
        return f"{title}\n{abstract}".strip()


def top5_pubkeys_for_queries(
    model,
    queries: list[dict],
    article_pubkeys: list[str],
) -> list[list[str]]:
    top5_preds: list[list[str]] = []

    # tqdm disabled to prevent context-window bloat in run.log
    for row in tqdm(queries, desc="Predicting", disable=True):
        top5_doc_indices = model.search(row["text"])[:5]
        top5_preds.append([article_pubkeys[idx] for idx in top5_doc_indices])

    return top5_preds


def main() -> None:
    split = "dev"
    collection = load_dataset(CHECKTHAT_DATASET, "collection", split="collection")

    article_pubkeys = [doc["pubkey"] for doc in collection]
    model = SparseRetriever()

    model.index(collection)

    all_results: list[dict] = []

    for lang in ["de", "fr", "en"]:
        queries = list(load_dataset(CHECKTHAT_DATASET, lang, split=split))
        num_queries = len(queries)

        top5_preds = top5_pubkeys_for_queries(model, queries, article_pubkeys)
        score = scorer(top5_preds=top5_preds, lang=lang, split=split)

        all_results.append({"lang": lang, "mrr5": score, "num_queries": num_queries})

        # Optional: Keep a minimal log for humans reading the run.log
        print(f"{lang.upper()} MRR@5: {score:.6f} (Queries: {num_queries})")

    # Calculate the query-weighted average (micro-average)
    total_queries = sum(r["num_queries"] for r in all_results)
    total_mrr_sum = sum(r["mrr5"] * r["num_queries"] for r in all_results)

    micro_avg_mrr5 = total_mrr_sum / total_queries if total_queries > 0 else 0.0

    # The distinct target line for the autoresearch agent
    print(f"FINAL_MICRO_AVG_MRR5: {micro_avg_mrr5:.6f}")


if __name__ == "__main__":
    main()
