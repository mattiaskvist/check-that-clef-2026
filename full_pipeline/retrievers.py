import re

import numpy as np
from deep_translator import GoogleTranslator
from deep_translator.exceptions import (
    NotValidLength,
    NotValidPayload,
    RequestError,
    TranslationNotFound,
)
from nltk.stem import LancasterStemmer
from rank_bm25 import BM25Plus

from .interfaces import BaseRetriever
from .utils import STOPWORDS


class BGEM3Retriever(BaseRetriever):
    def __init__(self, model_name: str = "BAAI/bge-m3", lora_id: str = None):
        # Heavy imports happen safely inside the cloud!
        import os

        import torch
        from peft import PeftModel
        from sentence_transformers import SentenceTransformer, util

        self.util = util
        self.torch = torch

        print(f"Loading Dense Retriever ({model_name})...")
        self.model = SentenceTransformer(model_name, device="cuda")

        if lora_id:
            print(f"Injecting LoRA adapters from {lora_id}...")
            hf_token = os.environ.get("HF_TOKEN")
            self.model[0].auto_model = PeftModel.from_pretrained(
                self.model[0].auto_model, lora_id, token=hf_token
            )
            self.model = self.model.to("cuda")

    def index(self, corpus: list[str]):
        print("Generating dense embeddings for the collection...")
        self.embeddings = self.model.encode_document(
            corpus, convert_to_tensor=True, show_progress_bar=True, device="cuda"
        )

    def search(self, query: str) -> list[int]:
        query_embedding = self.model.encode_query(query, convert_to_tensor=True)
        scores = self.util.cos_sim(query_embedding, self.embeddings)[0]
        return self.torch.argsort(scores, descending=True).tolist()


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

    def _translate_query(self, text: str, lang: str = "auto") -> str:
        """Helper function to translate non-English queries to English.

        Automatically detects language if lang='auto', otherwise uses the provided language code.

        Args:
            text (str): The original query text.
            lang (str): The language code of the query (e.g., 'en', 'de', 'fr', 'auto').

        Returns:
            str: The translated query text if translation was successful, otherwise the original text.
        """
        if lang == "en":
            return text

        translator = GoogleTranslator(source=lang, target="en")

        # Normalize text
        normalized_text = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", text)
        normalized_text = re.sub(
            r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U000024C2-\U0001F251]+",
            " ",
            normalized_text,
        )
        normalized_text = normalized_text.replace("#", " ")
        normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

        if not normalized_text:
            return text

        try:
            translated_text = translator.translate(text=normalized_text)
            original_terms: list[str] = []
            seen: set[str] = set()

            for tok in re.sub(r"[^\w\s]", " ", normalized_text.lower()).split():
                if tok in seen or tok in STOPWORDS or len(tok) < 6 or not tok.isalpha():
                    continue
                seen.add(tok)
                original_terms.append(tok)

            filtered_original = " ".join(original_terms)
            return (
                f"{translated_text} {filtered_original}".strip()
                if filtered_original
                else translated_text
            )
        except (TranslationNotFound, NotValidPayload, NotValidLength, RequestError):
            return normalized_text or text

    def search(self, query: str, lang: str = "auto") -> list[int]:
        """Translates the query if necessary, then performs BM25 search."""
        translated_query = self._translate_query(query, lang)
        tokenized_query = self.tokenize(translated_query)
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
        return f"{title} {title} {title} {title} {title} {title} {title} {title} {abstract} {authors} {venue}".strip()
