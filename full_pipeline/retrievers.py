import re

import numpy as np
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
    """A retriever that performs sparse retrieval.

    Args:
        BaseRetriever: The base retriever interface that this class implements.
    """

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
        title = (doc.get("title") or "").strip()
        abstract = (doc.get("abstract") or "").strip()
        # Repeat title to boost its importance
        return f"{title} {title} {title} {abstract}".strip()
