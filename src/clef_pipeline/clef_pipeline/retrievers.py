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
        import os

        import torch
        from peft import PeftModel
        from sentence_transformers import SentenceTransformer, util

        self.util = util
        self.torch = torch
        self.model_name = model_name
        self.lora_id = lora_id
        self.query_embeddings = None
        self._query_embeddings_by_name = {}

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

    @staticmethod
    def _texts_fingerprint(texts: list[str]) -> str:
        import hashlib

        digest = hashlib.sha256()
        for text in texts:
            encoded = text.encode("utf-8", errors="ignore")
            digest.update(len(encoded).to_bytes(8, "little", signed=False))
            digest.update(encoded)
        return digest.hexdigest()[:16]

    def _load_or_encode(
        self,
        texts: list[str],
        encode_fn,
        cache_path: str | None,
        label: str,
        force_recompute: bool = False,
    ):
        import os

        if cache_path and os.path.exists(cache_path) and not force_recompute:
            print(f"[cache hit] Loading {label} from {cache_path}")
            embs = self.torch.load(cache_path, map_location="cuda", weights_only=True)
            print(f"Loaded {embs.shape[0]} cached {label}.")
            return embs

        if force_recompute and cache_path and os.path.exists(cache_path):
            print(f"[cache bypass] Recomputing {label} from source texts.")

        print(f"Encoding {len(texts)} {label}...")
        embs = encode_fn(
            texts, convert_to_tensor=True, show_progress_bar=True, device="cuda"
        )

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            self.torch.save(embs, cache_path)
            print(f"[cache miss] Saved {label} to {cache_path}")

        return embs

    def _cache_path(
        self, cache_dir: str | None, filename: str, texts: list[str] | None = None
    ) -> str | None:
        import os

        if cache_dir:
            stem, extension = os.path.splitext(filename)
            if texts is not None:
                fingerprint = self._texts_fingerprint(texts)
                filename = f"{stem}-{fingerprint}{extension}"
            return os.path.join(cache_dir, self._cache_key(), filename)
        return None

    def index(
        self,
        corpus: list[str],
        cache_dir: str | None = None,
        force_recompute: bool = False,
    ):
        path = self._cache_path(cache_dir, "documents.pt", corpus)
        self.embeddings = self._load_or_encode(
            corpus,
            self.model.encode_document,
            path,
            "document embeddings",
            force_recompute=force_recompute,
        )

    def index_queries(
        self,
        queries: list[str],
        cache_dir: str | None = None,
        cache_name: str = "queries",
        force_recompute: bool = False,
    ):
        path = self._cache_path(cache_dir, f"{cache_name}.pt", queries)
        query_embeddings = self._load_or_encode(
            queries,
            self.model.encode_query,
            path,
            f"query embeddings ({cache_name})",
            force_recompute=force_recompute,
        )
        self.query_embeddings = query_embeddings
        self._query_embeddings_by_name[cache_name] = query_embeddings

    def search(self, query_idx: int, cache_name: str | None = None) -> list[int]:
        if cache_name is None:
            query_embeddings = self.query_embeddings
            if query_embeddings is None:
                raise ValueError(
                    "No query embeddings loaded. Call index_queries(...) first."
                )
        else:
            if cache_name not in self._query_embeddings_by_name:
                raise KeyError(f"No query cache named '{cache_name}' is available.")
            query_embeddings = self._query_embeddings_by_name[cache_name]

        scores = self.util.cos_sim(query_embeddings[query_idx], self.embeddings)[0]
        return self.torch.argsort(scores, descending=True).tolist()


class HarrierRetriever(BaseRetriever):
    DEFAULT_QUERY_PROMPT = (
        "Instruct: Retrieve the implicitly referenced scientific article\nQuery: "
    )

    def __init__(
        self, model_name: str = "microsoft/harrier-oss-v1-27b", batch_size: int = 2
    ):
        import torch
        from sentence_transformers import SentenceTransformer, util

        self.util = util
        self.torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_embeddings = None
        self._query_embeddings_by_name = {}
        self.prompt = self.DEFAULT_QUERY_PROMPT

        print(f"Loading Dense Retriever ({model_name})...")
        self.model = SentenceTransformer(
            model_name, device="cuda", model_kwargs={"dtype": "auto"}
        )

    def _cache_key(self) -> str:
        return self.model_name.replace("/", "--")

    @staticmethod
    def _texts_fingerprint(texts: list[str]) -> str:
        import hashlib

        digest = hashlib.sha256()
        for text in texts:
            encoded = text.encode("utf-8", errors="ignore")
            digest.update(len(encoded).to_bytes(8, "little", signed=False))
            digest.update(encoded)
        return digest.hexdigest()[:16]

    def _cache_path(
        self, cache_dir: str | None, filename: str, texts: list[str] | None = None
    ) -> str | None:
        import os

        if cache_dir:
            stem, extension = os.path.splitext(filename)
            if texts is not None:
                fingerprint = self._texts_fingerprint(texts)
                filename = f"{stem}-{fingerprint}{extension}"
            return os.path.join(cache_dir, self._cache_key(), filename)
        return None

    def _load_or_encode(
        self,
        texts: list[str],
        cache_path: str | None,
        label: str,
        force_recompute: bool = False,
        **encode_kwargs,
    ):
        import os

        if cache_path and os.path.exists(cache_path) and not force_recompute:
            print(f"[cache hit] Loading {label} from {cache_path}")
            embs = self.torch.load(cache_path, map_location="cuda", weights_only=True)
            print(f"Loaded {embs.shape[0]} cached {label}.")
            return embs

        if force_recompute and cache_path and os.path.exists(cache_path):
            print(f"[cache bypass] Recomputing {label} from source texts.")

        print(f"Encoding {len(texts)} {label}...")
        embs = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            device="cuda",
            batch_size=self.batch_size,
            **encode_kwargs,
        )

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            self.torch.save(embs, cache_path)
            print(f"[cache miss] Saved {label} to {cache_path}")

        return embs

    def index(
        self,
        corpus: list[str],
        cache_dir: str | None = None,
        force_recompute: bool = False,
    ):
        path = self._cache_path(cache_dir, "documents.pt", corpus)
        self.embeddings = self._load_or_encode(
            corpus,
            path,
            "document embeddings",
            force_recompute=force_recompute,
        )

    def index_queries(
        self,
        queries: list[str],
        cache_dir: str | None = None,
        cache_name: str = "queries",
        force_recompute: bool = False,
    ):
        path = self._cache_path(cache_dir, f"{cache_name}.pt", queries)
        query_embeddings = self._load_or_encode(
            queries,
            path,
            f"query embeddings ({cache_name})",
            force_recompute=force_recompute,
            prompt=getattr(self, "prompt", self.DEFAULT_QUERY_PROMPT),
        )
        self.query_embeddings = query_embeddings
        self._query_embeddings_by_name[cache_name] = query_embeddings

    def unload_model(self):
        """Free the embedding model from GPU. Computed embeddings are kept."""
        import gc

        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()
        print("Embedding model unloaded, GPU memory freed.")

    def search(self, query_idx: int, cache_name: str | None = None) -> list[int]:
        if cache_name is None:
            query_embeddings = self.query_embeddings
            if query_embeddings is None:
                raise ValueError(
                    "No query embeddings loaded. Call index_queries(...) first."
                )
        else:
            query_sets = getattr(self, "_query_embeddings_by_name", {})
            if cache_name not in query_sets:
                raise KeyError(f"No query cache named '{cache_name}' is available.")
            query_embeddings = query_sets[cache_name]

        scores = self.util.cos_sim(query_embeddings[query_idx], self.embeddings)[0]
        return self.torch.argsort(scores, descending=True).tolist()


class SparseRetriever(BaseRetriever):
    """A retriever that performs sparse retrieval."""

    def __init__(self):
        self.bm25_model = None
        self.bm25_k1 = 2.5
        self.bm25_b = 0.85
        self.stemmer = LancasterStemmer()
        self._indexed_corpus_fingerprint = None
        self._query_rankings_by_name = {}
        self._query_scores_by_name = {}

    def _cache_key(self) -> str:
        return (
            f"bm25plus-k1_{self.bm25_k1:.2f}-b_{self.bm25_b:.2f}"
            "-stem_lancaster-bigrams_1-translate_v1"
        )

    @staticmethod
    def _texts_fingerprint(texts: list[str]) -> str:
        import hashlib

        digest = hashlib.sha256()
        for text in texts:
            encoded = text.encode("utf-8", errors="ignore")
            digest.update(len(encoded).to_bytes(8, "little", signed=False))
            digest.update(encoded)
        return digest.hexdigest()[:16]

    def _cache_path(
        self,
        cache_dir: str | None,
        cache_name: str,
        queries: list[str],
        lang: str,
        top_k: int | None,
    ) -> str | None:
        import os

        if cache_dir is None:
            return None

        if self._indexed_corpus_fingerprint is None:
            raise ValueError("SparseRetriever must be indexed before caching queries.")

        query_fingerprint = self._texts_fingerprint(queries)
        top_k_label = "all" if top_k is None else str(top_k)
        filename = (
            f"{cache_name}-{lang}-top{top_k_label}"
            f"-{self._indexed_corpus_fingerprint}-{query_fingerprint}.npz"
        )
        return os.path.join(cache_dir, "sparse", self._cache_key(), filename)

    def _score_query(
        self, query: str, lang: str = "auto", top_k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.bm25_model is None:
            raise ValueError("SparseRetriever is not indexed. Call index(...) first.")

        translated_query = self._translate_query(query, lang)
        tokenized_query = self.tokenize(translated_query)
        scores = np.asarray(
            self.bm25_model.get_scores(tokenized_query), dtype=np.float32
        )
        ranked_indices = np.argsort(scores)[::-1]
        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]
        ranked_scores = scores[ranked_indices]
        return ranked_indices.astype(np.int32), ranked_scores.astype(np.float32)

    def _get_cached_query(
        self, query_idx: int, cache_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if cache_name not in self._query_rankings_by_name:
            raise KeyError(f"No sparse query cache named '{cache_name}' is available.")

        rankings = self._query_rankings_by_name[cache_name]
        scores = self._query_scores_by_name[cache_name]

        if query_idx < 0 or query_idx >= len(rankings):
            raise IndexError(
                f"Query index {query_idx} is out of bounds for cache '{cache_name}'."
            )

        return np.asarray(rankings[query_idx]), np.asarray(
            scores[query_idx], dtype=np.float32
        )

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
        self.bm25_model = BM25Plus(tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b)
        self._indexed_corpus_fingerprint = self._texts_fingerprint(corpus)
        self._query_rankings_by_name.clear()
        self._query_scores_by_name.clear()

    def index_queries(
        self,
        queries: list[str],
        lang: str = "auto",
        cache_dir: str | None = None,
        cache_name: str = "queries",
        top_k: int | None = 2000,
        force_recompute: bool = False,
    ):
        import os

        if self.bm25_model is None:
            raise ValueError("SparseRetriever is not indexed. Call index(...) first.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0 when provided.")

        cache_path = self._cache_path(cache_dir, cache_name, queries, lang, top_k)
        if cache_path and os.path.exists(cache_path) and not force_recompute:
            print(
                f"[cache hit] Loading sparse query cache ({cache_name}) from {cache_path}"
            )
            with np.load(cache_path, allow_pickle=False) as cached:
                rankings = cached["rankings"]
                scores = cached["scores"]
            self._query_rankings_by_name[cache_name] = rankings
            self._query_scores_by_name[cache_name] = scores
            print(f"Loaded {len(rankings)} cached sparse query results ({cache_name}).")
            return

        if force_recompute and cache_path and os.path.exists(cache_path):
            print(f"[cache bypass] Recomputing sparse query cache ({cache_name}).")

        corpus_size = len(getattr(self.bm25_model, "doc_len", []))
        effective_top_k = corpus_size if top_k is None else min(top_k, corpus_size)
        rankings = np.empty((len(queries), effective_top_k), dtype=np.int32)
        scores = np.empty((len(queries), effective_top_k), dtype=np.float32)

        for query_idx, query_text in enumerate(queries):
            ranked_indices, ranked_scores = self._score_query(
                query_text, lang=lang, top_k=effective_top_k
            )
            rankings[query_idx] = ranked_indices
            scores[query_idx] = ranked_scores

        self._query_rankings_by_name[cache_name] = rankings
        self._query_scores_by_name[cache_name] = scores

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, rankings=rankings, scores=scores)
            print(
                f"[cache miss] Saved sparse query cache ({cache_name}) to {cache_path}"
            )

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

    def search_with_scores(
        self,
        query: str | int,
        lang: str = "auto",
        cache_name: str | None = None,
    ) -> tuple[list[int], list[float]]:
        if cache_name is not None:
            if not isinstance(query, int):
                raise TypeError("Cached sparse lookup requires query index (int).")
            ranked_indices, ranked_scores = self._get_cached_query(query, cache_name)
            return ranked_indices.tolist(), ranked_scores.tolist()

        if not isinstance(query, str):
            raise TypeError("Query must be a string when cache_name is not provided.")

        ranked_indices, ranked_scores = self._score_query(query, lang=lang, top_k=None)
        return ranked_indices.tolist(), ranked_scores.tolist()

    def search(
        self, query: str | int, lang: str = "auto", cache_name: str | None = None
    ) -> list[int]:
        """Return sparse-ranked document ids for a query text or cached query index."""
        ranked_indices, _ = self.search_with_scores(
            query, lang=lang, cache_name=cache_name
        )
        return ranked_indices

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
