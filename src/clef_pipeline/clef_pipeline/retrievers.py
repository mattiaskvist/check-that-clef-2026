"""Dense and sparse retriever implementations for pipeline stages."""

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

from .interfaces import BaseRetriever
from .utils import STOPWORDS


class BGEM3Retriever(BaseRetriever):
    """Dense retriever using SentenceTransformer BGE-M3 embeddings."""

    def __init__(self, model_name: str = "BAAI/bge-m3", lora_id: str = None):
        """Load dense retriever model and optional LoRA adapters.

        Args:
            model_name: Base embedding model id.
            lora_id: Optional LoRA adapter id to inject into the base model.
        """
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
        """Build cache namespace identifier for model and adapter settings."""
        key = self.model_name.replace("/", "--")
        if self.lora_id:
            key += f"+{self.lora_id.replace('/', '--')}"
        return key

    @staticmethod
    def _texts_fingerprint(texts: list[str]) -> str:
        """Compute a deterministic short fingerprint for text collections."""
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
        """Load cached embeddings or encode and persist them.

        Args:
            texts: Input texts to encode.
            encode_fn: Callable used for model encoding.
            cache_path: Optional on-disk cache location.
            label: Human-readable label for logging.
            force_recompute: Bypass existing cache when true.

        Returns:
            Embedding tensor loaded from cache or produced by encoding.
        """
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
        """Build cache file path for embeddings.

        Args:
            cache_dir: Root cache directory or ``None`` to disable caching.
            filename: Base cache file name.
            texts: Optional source texts for fingerprinted naming.

        Returns:
            Cache path or ``None`` when caching is disabled.
        """
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
        """Index document texts by computing dense document embeddings.

        Args:
            corpus: Document texts.
            cache_dir: Optional cache directory for embedding tensors.
            force_recompute: Recompute embeddings even if cache exists.
        """
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
        """Index query texts and store embeddings by cache name.

        Args:
            queries: Query texts to encode.
            cache_dir: Optional cache directory for embedding tensors.
            cache_name: Query embedding cache namespace.
            force_recompute: Recompute embeddings even if cache exists.
        """
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
        """Return dense ranking for one indexed query.

        Args:
            query_idx: Index of the query embedding.
            cache_name: Optional query cache name to read from.

        Returns:
            Ranked document indices.

        Raises:
            ValueError: If default query embeddings are unavailable.
            KeyError: If the requested cache name is not indexed.
        """
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
    """Dense retriever using Microsoft Harrier embedding models."""

    DEFAULT_QUERY_PROMPT = (
        "Instruct: Retrieve the implicitly referenced scientific article\nQuery: "
    )

    def __init__(
        self, model_name: str = "microsoft/harrier-oss-v1-27b", batch_size: int = 2
    ):
        """Load Harrier embedding model and runtime settings.

        Args:
            model_name: Harrier model id.
            batch_size: Embedding batch size for encode calls.
        """
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
        """Build cache namespace identifier for model settings."""
        return self.model_name.replace("/", "--")

    @staticmethod
    def _texts_fingerprint(texts: list[str]) -> str:
        """Compute a deterministic short fingerprint for text collections."""
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
        """Build cache file path for embeddings."""
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
        """Load cached embeddings or encode and persist them.

        Args:
            texts: Input texts to encode.
            cache_path: Optional on-disk cache location.
            label: Human-readable label for logging.
            force_recompute: Bypass existing cache when true.
            **encode_kwargs: Extra kwargs passed to ``SentenceTransformer.encode``.

        Returns:
            Embedding tensor loaded from cache or produced by encoding.
        """
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
        """Index document texts by computing dense document embeddings."""
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
        """Index query texts and store embeddings by cache name."""
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

    def search_with_scores(
        self, query_idx: int, cache_name: str | None = None
    ) -> tuple[list[int], list[float]]:
        """Return (ranked_doc_ids, scores) for a cached query.

        Mirrors SparseRetriever.search_with_scores(). The scores list is
        indexed by doc_id (not by rank), matching the dense score array layout
        needed by FeatureGenerator.
        """
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
        ranked = self.torch.argsort(scores, descending=True).tolist()
        return ranked, scores.cpu().numpy().tolist()

    def search(self, query_idx: int, cache_name: str | None = None) -> list[int]:
        """Return dense ranking for one indexed query.

        Args:
            query_idx: Index of the query embedding.
            cache_name: Optional query cache name to read from.

        Returns:
            Ranked document indices.

        Raises:
            ValueError: If default query embeddings are unavailable.
            KeyError: If the requested cache name is not indexed.
        """
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

    def __init__(
        self,
        k1: float = 2.5,
        b: float = 0.85,
        use_bigrams: bool = True,
        use_translation: bool = True,
    ):
        """Initialize sparse retriever parameters and caches."""
        self.bm25_model = None
        self.bm25_k1 = k1
        self.bm25_b = b
        self.use_bigrams = use_bigrams
        self.use_translation = use_translation
        self.stemmer = LancasterStemmer()
        self.diffusion_steps = 2
        self.diffusion_decay = 0.65
        self.diff_neighbors = 6
        self.prf_docs = 7
        self.prf_terms = 8
        self.prf_weight = 0.85
        self.window_size = 5
        self._docs_tokens = None
        self._term_graph = None
        self._indexed_corpus_fingerprint = None
        self._query_rankings_by_name = {}
        self._query_scores_by_name = {}

    @staticmethod
    def _extract_german_author_tokens(text: str) -> list[str]:
        """Extract likely person-name tokens from German queries."""
        if not text:
            return []

        candidates = re.findall(
            r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß-]{2,}\b",
            text,
            flags=re.UNICODE,
        )
        tokens: list[str] = []
        seen: set[str] = set()
        for token in candidates:
            normalized = token.strip("-").lower()
            if (
                not normalized
                or normalized in seen
                or normalized in STOPWORDS
                or len(normalized) <= 1
            ):
                continue
            seen.add(normalized)
            tokens.append(normalized)
        return tokens

    @staticmethod
    def _author_field_tokens(text: str) -> list[str]:
        """Tokenize author strings into dedicated index tokens.

        Prefixing ensures author matches only happen for queries that also emit
        `author_...` tokens (German-only heuristic).
        """
        if not text:
            return []
        raw = re.sub(r"[^\w\s-]", " ", text.lower())
        toks: list[str] = []
        seen: set[str] = set()
        for tok in raw.split():
            tok = tok.strip("-")
            if not tok or tok in seen or tok in STOPWORDS or len(tok) <= 1:
                continue
            seen.add(tok)
            toks.append(f"author_{tok}")
        return toks

    def _cache_key(self) -> str:
        """Build cache namespace identifier for sparse retrieval settings."""
        use_bigrams = getattr(self, "use_bigrams", True)
        use_translation = getattr(self, "use_translation", True)
        bigrams_val = 1 if use_bigrams else 0
        trans_val = "v1" if use_translation else "none"
        return (
            f"bm25plus-k1_{self.bm25_k1:.2f}-b_{self.bm25_b:.2f}"
            f"-stem_lancaster-bigrams_{bigrams_val}-translate_{trans_val}"
        )

    @staticmethod
    def _texts_fingerprint(texts: list[str]) -> str:
        """Compute a deterministic short fingerprint for text collections."""
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
        """Build cache path for sparse query rankings and scores.

        Args:
            cache_dir: Root cache directory.
            cache_name: Query cache namespace.
            queries: Query texts.
            lang: Language key used for cache identity.
            top_k: Cached cutoff depth.

        Returns:
            Cache path or ``None`` if caching is disabled.

        Raises:
            ValueError: If called before the retriever has been indexed.
        """
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
        """Score one query with BM25+ and return ranked ids and scores.

        Args:
            query: Query text.
            lang: Query language code or ``auto`` for auto-detection.
            top_k: Optional top-k truncation.

        Returns:
            Tuple of ranked document indices and aligned scores.

        Raises:
            ValueError: If retriever was not indexed.
        """
        if self.bm25_model is None:
            raise ValueError("SparseRetriever is not indexed. Call index(...) first.")

        if self.use_translation:
            translated_query = self._translate_query(query, lang)
        else:
            translated_query = query

        tokenized_query = self.tokenize(translated_query)
        if lang == "de":
            tokenized_query.extend(
                f"author_{tok}" for tok in self._extract_german_author_tokens(query)
            )
        scores_dict = self.bm25_model.get_scores(tokenized_query)
        if not scores_dict:
            if top_k is None:
                top_k = 0
            return np.empty((top_k,), dtype=np.int32), np.zeros((top_k,), dtype=np.float32)

        if lang == "en" and self._term_graph is not None:
            expanded = self._diffusion_expand(tokenized_query, self._term_graph)
            for term, weight in expanded.items():
                postings = self.bm25_model.index.get(term)
                if not postings:
                    continue
                for doc_id, _ in postings:
                    scores_dict[doc_id] += weight

            docs_tokens = self._docs_tokens or []
            top_docs = self._top_docs(scores_dict, k=self.prf_docs)
            if top_docs and docs_tokens:
                from collections import Counter

                counter = Counter()
                for doc_id in top_docs:
                    if 0 <= doc_id < len(docs_tokens):
                        for tok in docs_tokens[doc_id]:
                            counter[tok] += 1

                total = float(sum(counter.values())) + 1e-9
                for term, count in counter.most_common(self.prf_terms):
                    postings = self.bm25_model.index.get(term)
                    if not postings:
                        continue
                    weight = (count / total)
                    for doc_id, _ in postings:
                        scores_dict[doc_id] += self.prf_weight * weight

        mx = max(scores_dict.values()) + 1e-9
        for doc_id in list(scores_dict.keys()):
            scores_dict[doc_id] /= mx

        ranked_indices, ranked_scores = self._rank_and_pad(scores_dict, top_k=top_k)
        return ranked_indices, ranked_scores

    def _get_cached_query(
        self, query_idx: int, cache_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fetch ranked ids and scores for one cached query.

        Args:
            query_idx: Query position in cache arrays.
            cache_name: Cached query namespace.

        Returns:
            Ranked ids and scores for the selected query.

        Raises:
            KeyError: If cache name does not exist.
            IndexError: If query index is out of range.
        """
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

    def tokenize(self, text: str, add_bigrams: bool | None = None) -> list[str]:
        """Tokenize text with punctuation, stopword removal, stemming, and optional bigrams."""
        if add_bigrams is None:
            add_bigrams = self.use_bigrams
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = text.split()
        unigrams = [
            self.stemmer.stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1
        ]
        if add_bigrams and len(unigrams) >= 2:
            bigrams = [
                f"{unigrams[i]}_{unigrams[i + 1]}" for i in range(len(unigrams) - 1)
            ]
            return unigrams + bigrams
        return unigrams

    def index(self, collection: list[dict]):
        """Index a document collection into a BM25+ model.

        Args:
            collection: Document dictionaries with article metadata.
        """
        from collections import defaultdict

        class FastBM25:
            def __init__(self, docs: list[list[str]], k1: float, b: float):
                self.k1 = float(k1)
                self.b = float(b)
                self.N = len(docs)
                self.doc_len = np.asarray([len(d) for d in docs], dtype=np.float32)
                self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
                self.index = defaultdict(list)
                self.df = defaultdict(int)

                for doc_id, doc in enumerate(docs):
                    freqs = defaultdict(int)
                    for tok in doc:
                        freqs[tok] += 1
                    for tok, tf in freqs.items():
                        self.index[tok].append((doc_id, tf))
                        self.df[tok] += 1

                self.idf = {
                    tok: float(np.log(1 + (self.N - df + 0.5) / (df + 0.5)))
                    for tok, df in self.df.items()
                }

            def get_scores(self, query: list[str]):
                scores = defaultdict(float)
                if self.N == 0:
                    return scores

                for tok in query:
                    postings = self.index.get(tok)
                    if not postings:
                        continue
                    idf = self.idf.get(tok, 0.0)
                    for doc_id, tf in postings:
                        denom = tf + self.k1 * (
                            1 - self.b + self.b * (self.doc_len[doc_id] / self.avgdl)
                        )
                        scores[doc_id] += idf * (tf * (self.k1 + 1) / denom)
                return scores

        corpus = [self.document_to_text(doc) for doc in collection]
        tokenized_corpus = [self.tokenize(text) for text in corpus]
        self.bm25_model = FastBM25(tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b)
        self._docs_tokens = tokenized_corpus
        self._term_graph = self._build_term_graph(tokenized_corpus)
        self._indexed_corpus_fingerprint = self._texts_fingerprint(corpus)
        self._query_rankings_by_name.clear()
        self._query_scores_by_name.clear()

    def _build_term_graph(self, docs: list[list[str]]):
        from collections import Counter, defaultdict

        term_graph = defaultdict(Counter)
        window_size = int(self.window_size)

        for doc in docs:
            for i, term in enumerate(doc):
                window = doc[i + 1 : i + 1 + window_size]
                for neighbor in window:
                    if term == neighbor:
                        continue
                    term_graph[term][neighbor] += 1
                    term_graph[neighbor][term] += 1
        return term_graph

    def _diffusion_expand(self, tokens: list[str], term_graph):
        from collections import Counter

        weights = Counter({t: 1.0 for t in tokens})

        for _ in range(int(self.diffusion_steps)):
            new_weights = Counter()
            for term, weight in weights.items():
                neighbors = term_graph.get(term)
                if not neighbors:
                    continue
                total = float(sum(neighbors.values())) + 1e-9
                for neighbor, count in neighbors.most_common(int(self.diff_neighbors)):
                    new_weights[neighbor] += weight * (count / total) * float(
                        self.diffusion_decay
                    )
            weights.update(new_weights)

        return weights

    @staticmethod
    def _top_docs(scores_dict, k: int) -> list[int]:
        if k <= 0 or not scores_dict:
            return []
        doc_ids = np.fromiter(scores_dict.keys(), dtype=np.int32)
        vals = np.fromiter(scores_dict.values(), dtype=np.float32)
        k = min(int(k), len(vals))
        top_idx = np.argpartition(vals, -k)[-k:]
        top_idx = top_idx[np.argsort(vals[top_idx])[::-1]]
        return doc_ids[top_idx].tolist()

    def _rank_and_pad(
        self, scores_dict, top_k: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        corpus_size = len(getattr(getattr(self, "bm25_model", None), "doc_len", []))
        if corpus_size <= 0:
            corpus_size = len(getattr(self, "_docs_tokens", []) or [])

        if top_k is None:
            top_k = corpus_size
        top_k = min(int(top_k), int(corpus_size))

        doc_ids = np.fromiter(scores_dict.keys(), dtype=np.int32)
        vals = np.fromiter(scores_dict.values(), dtype=np.float32)
        if len(vals) == 0:
            return np.empty((top_k,), dtype=np.int32), np.zeros((top_k,), dtype=np.float32)

        k = min(top_k, len(vals))
        top_idx = np.argpartition(vals, -k)[-k:]
        top_idx = top_idx[np.argsort(vals[top_idx])[::-1]]
        ranked_ids = doc_ids[top_idx].astype(np.int32)
        ranked_scores = vals[top_idx].astype(np.float32)

        if len(ranked_ids) < top_k:
            used = set(int(i) for i in ranked_ids.tolist())
            needed = top_k - len(ranked_ids)
            filler = []
            for doc_id in range(int(corpus_size)):
                if doc_id not in used:
                    filler.append(doc_id)
                    if len(filler) >= needed:
                        break
            if filler:
                ranked_ids = np.concatenate([ranked_ids, np.asarray(filler, dtype=np.int32)])
                ranked_scores = np.concatenate(
                    [ranked_scores, np.zeros((len(filler),), dtype=np.float32)]
                )

        return ranked_ids, ranked_scores

    def index_queries(
        self,
        queries: list[str],
        lang: str = "auto",
        cache_dir: str | None = None,
        cache_name: str = "queries",
        top_k: int | None = 2000,
        force_recompute: bool = False,
    ):
        """Precompute sparse rankings for a query set and optionally cache them.

        Args:
            queries: Query texts to score.
            lang: Query language code or ``auto``.
            cache_dir: Optional cache directory.
            cache_name: Cache namespace for query rankings.
            top_k: Maximum number of ranked documents per query.
            force_recompute: Recompute even if cached arrays exist.

        Raises:
            ValueError: If retriever is not indexed or ``top_k`` is invalid.
        """
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
        """Return sparse-ranked document ids and scores for a query.

        Args:
            query: Query text, or query index when ``cache_name`` is provided.
            lang: Query language code or ``auto``.
            cache_name: Optional cache namespace for indexed queries.

        Returns:
            Ranked ids and aligned relevance scores.

        Raises:
            TypeError: If query type does not match cache usage mode.
        """

        def _scores_by_doc_id(
            ranked_indices: np.ndarray, ranked_scores: np.ndarray
        ) -> list[float]:
            corpus_size = len(getattr(getattr(self, "bm25_model", None), "doc_len", []))
            if corpus_size <= 0 and len(ranked_indices) > 0:
                corpus_size = int(np.max(ranked_indices)) + 1
            full_scores = np.zeros(corpus_size, dtype=np.float32)
            full_scores[ranked_indices] = ranked_scores
            return full_scores.tolist()

        if cache_name is not None:
            if not isinstance(query, int):
                raise TypeError("Cached sparse lookup requires query index (int).")
            ranked_indices, ranked_scores = self._get_cached_query(query, cache_name)
            return ranked_indices.tolist(), _scores_by_doc_id(
                ranked_indices, ranked_scores
            )

        if not isinstance(query, str):
            raise TypeError("Query must be a string when cache_name is not provided.")

        ranked_indices, ranked_scores = self._score_query(query, lang=lang, top_k=None)
        return ranked_indices.tolist(), _scores_by_doc_id(ranked_indices, ranked_scores)

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
        venue = str(doc.get("venue") or "").strip()
        authors_raw = doc.get("authors") or ""
        authors = (
            " ".join(str(part) for part in authors_raw)
            if isinstance(authors_raw, list)
            else str(authors_raw)
        ).strip()

        # Keep author and venue fields in the sparse index to support
        # author-centric queries, but gate author matching behind `author_...`
        # tokens so it effectively only applies for German.
        author_tokens = self._author_field_tokens(authors) if authors else []
        return " ".join(
            [title, title, title, title, title, abstract, *author_tokens, venue, venue]
        ).strip()
