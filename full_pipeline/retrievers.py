import re

import numpy as np
from nltk.stem import LancasterStemmer
from rank_bm25 import BM25Plus

from .interfaces import BaseRetriever
from .utils import STOPWORDS


class BGEM3Retriever(BaseRetriever):
    def __init__(self, model_name: str = "BAAI/bge-m3", lora_id: str = None):
        import os

        import torch
        from peft import PeftModel
        from sentence_transformers import SentenceTransformer, util

        self.util = util
        self.torch = torch
        self.model_name = model_name
        self.lora_id = lora_id

        print(f"Loading Dense Retriever ({model_name})...")
        self.model = SentenceTransformer(model_name, device="cuda")

        if lora_id:
            print(f"Injecting LoRA adapters from {lora_id}...")
            hf_token = os.environ.get("HF_TOKEN")
            self.model[0].auto_model = PeftModel.from_pretrained(
                self.model[0].auto_model, lora_id, token=hf_token
            )
            self.model = self.model.to("cuda")

    def _cache_key(self) -> str:
        key = self.model_name.replace("/", "--")
        if self.lora_id:
            key += f"+{self.lora_id.replace('/', '--')}"
        return key

    def _load_or_encode(self, texts: list[str], encode_fn, cache_path: str | None, label: str):
        import os

        if cache_path and os.path.exists(cache_path):
            print(f"[cache hit] Loading {label} from {cache_path}")
            embs = self.torch.load(cache_path, map_location="cuda", weights_only=True)
            print(f"Loaded {embs.shape[0]} cached {label}.")
            return embs

        print(f"Encoding {len(texts)} {label}...")
        embs = encode_fn(texts, convert_to_tensor=True, show_progress_bar=True, device="cuda")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            self.torch.save(embs, cache_path)
            print(f"[cache miss] Saved {label} to {cache_path}")

        return embs

    def _cache_path(self, cache_dir: str | None, filename: str) -> str | None:
        import os
        if cache_dir:
            return os.path.join(cache_dir, self._cache_key(), filename)
        return None

    def index(self, corpus: list[str], cache_dir: str | None = None):
        path = self._cache_path(cache_dir, "documents.pt")
        self.embeddings = self._load_or_encode(corpus, self.model.encode_document, path, "document embeddings")

    def index_queries(self, queries: list[str], cache_dir: str | None = None, cache_name: str = "queries"):
        path = self._cache_path(cache_dir, f"{cache_name}.pt")
        self.query_embeddings = self._load_or_encode(queries, self.model.encode_query, path, f"query embeddings ({cache_name})")

    def search(self, query_idx: int) -> list[int]:
        scores = self.util.cos_sim(self.query_embeddings[query_idx], self.embeddings)[0]
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
