"""Dense and sparse retriever implementations for pipeline stages."""

import re

import numpy as np
from nltk.stem import LancasterStemmer, SnowballStemmer
from rank_bm25 import BM25Plus

from .interfaces import BaseRetriever
from .translation_cache import TranslationCache, translate_batch
from .utils import STOPWORDS


DEFAULT_FIELD_WEIGHTS: dict[str, int] = {
    "title": 3,
    "abstract": 1,
    "authors": 1,
    "venue": 1,
}


def _select_stemmer(language: str):
    """Return an NLTK stemmer suitable for ``language``.

    German and French get their own Snowball stemmer instead of the
    English-only Lancaster stemmer that the original implementation used
    for all languages after translation. Falling back to Lancaster for
    English preserves existing behavior when tokenizing already-English
    (or post-translation) text.
    """
    lang = (language or "en").lower()
    if lang.startswith("de"):
        return SnowballStemmer("german")
    if lang.startswith("fr"):
        return SnowballStemmer("french")
    return LancasterStemmer()


class BGEM3Retriever(BaseRetriever):
    """Dense retriever using SentenceTransformer BGE-M3 embeddings."""

    def __init__(self, model_name: str = "BAAI/bge-m3", lora_id: str = None):
        """Configure dense retriever; weights are loaded lazily on cache miss.

        Args:
            model_name: Base embedding model id.
            lora_id: Optional LoRA adapter id to inject into the base model.
        """
        import torch
        from sentence_transformers import util

        self.util = util
        self.torch = torch
        self.model_name = model_name
        self.lora_id = lora_id
        self.query_embeddings = None
        self._query_embeddings_by_name = {}
        # Lazy: only instantiated when we actually need to encode (cache miss).
        # Warm-cache evaluation runs skip the multi-gigabyte weight download.
        self.model = None

    def _ensure_model_loaded(self):
        """Load base encoder + optional LoRA weights on first encode call."""
        if self.model is not None:
            return
        import os

        from peft import PeftModel
        from sentence_transformers import SentenceTransformer

        print(f"Loading Dense Retriever ({self.model_name})...")
        self.model = SentenceTransformer(self.model_name, device="cuda")
        if self.lora_id:
            print(f"Injecting LoRA adapters from {self.lora_id}...")
            hf_token = os.environ.get("HF_TOKEN")
            self.model[0].auto_model = PeftModel.from_pretrained(
                self.model[0].auto_model, self.lora_id, token=hf_token
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
        encode_attr: str,
        cache_path: str | None,
        label: str,
        force_recompute: bool = False,
    ):
        """Load cached embeddings or encode and persist them.

        Args:
            texts: Input texts to encode.
            encode_attr: Name of the model method to call for encoding
                (e.g. ``"encode_document"``). Resolved lazily after weights
                are loaded so cache hits never trigger a model load.
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

        self._ensure_model_loaded()
        encode_fn = getattr(self.model, encode_attr)
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
            "encode_document",
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
            "encode_query",
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
        self,
        model_name: str = "microsoft/harrier-oss-v1-27b",
        batch_size: int = 1,
        adapter_path: str | None = None,
        max_seq_length: int = 1024,
    ):
        """Load Harrier embedding model and runtime settings.

        Args:
            model_name: Harrier model id.
            batch_size: Embedding batch size for encode calls. Defaults to 1
                because Harrier-27B + sliding-window attention masks can push
                an A100-80GB over the edge for any batch > 1; document
                encoding is cached so the one-time cost is tolerable.
            adapter_path: Optional path to a PEFT LoRA adapter directory. When
                set, the adapter is loaded on top of the base model. The cache
                key incorporates the adapter's basename so fine-tuned and base
                embeddings don't clash in the cache volume.
            max_seq_length: Hard cap on tokenized sequence length. Harrier
                inherits Gemma3's 131k-token max_position_embeddings by
                default, which makes attention O(seq_len^2) OOM on long
                abstracts/author lists. 1024 captures title + abstract body
                for essentially all scientific papers and keeps attention
                memory bounded (~256 MB per layer at batch=1).
        """
        import torch
        from sentence_transformers import util

        self.util = util
        self.torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.adapter_path = adapter_path
        self.max_seq_length = int(max_seq_length)
        self.query_embeddings = None
        self._query_embeddings_by_name = {}
        self.prompt = self.DEFAULT_QUERY_PROMPT
        # Lazy: weights are only loaded when a cache miss forces encoding.
        # Warm-cache evaluation runs avoid the multi-minute 27B model load.
        self.model = None

    def _ensure_model_loaded(self) -> None:
        """Load SentenceTransformer weights and optional adapter on first use."""
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer

        print(f"Loading Dense Retriever ({self.model_name})...")
        # NOTE: transformers<4.55 uses ``torch_dtype``; 4.55+ renamed the
        # argument to ``dtype``. We're pinned to <4.55 to avoid the vmap
        # sliding-window mask OOM, so keep the old name here.
        self.model = SentenceTransformer(
            self.model_name, device="cuda", model_kwargs={"torch_dtype": "auto"}
        )
        self.model.max_seq_length = int(self.max_seq_length)
        if self.adapter_path:
            self._attach_adapter(self.adapter_path)

    def _attach_adapter(self, adapter_path: str) -> None:
        """Wrap the underlying transformer with a PEFT adapter."""
        from peft import PeftModel

        print(f"Attaching LoRA adapter from {adapter_path}...")
        self.model[0].auto_model = PeftModel.from_pretrained(
            self.model[0].auto_model, adapter_path
        )

    def _cache_key(self) -> str:
        """Build cache namespace identifier for model settings.

        ``max_seq_length`` is part of the key because truncation at encode
        time materially changes the document embedding for long articles;
        a 1024-token truncation produces different vectors than 8192.
        """
        base = self.model_name.replace("/", "--")
        max_seq_length = getattr(self, "max_seq_length", None) or int(
            getattr(getattr(self, "model", None), "max_seq_length", 0) or 0
        )
        if max_seq_length:
            base = f"{base}--msl{int(max_seq_length)}"
        adapter_path = getattr(self, "adapter_path", None)
        if adapter_path:
            import os

            tag = os.path.basename(os.path.normpath(adapter_path))
            return f"{base}--adapter-{tag}"
        return base

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

        self._ensure_model_loaded()
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

        if self.model is None:
            return
        del self.model
        self.model = None
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
    """BM25+ retriever with per-language analyzers and field-weighted indexing.

    Improvements vs. the original implementation:

    - Per-query ``GoogleTranslator`` calls are replaced with a single batched
      translation pass at indexing time, persisted in a disk cache (see
      ``translation_cache.py``). Results are deterministic and cacheable.
    - Stemmer is selected by the *source* language: English keeps Lancaster,
      German/French use their Snowball stemmers rather than getting run
      through a stemmer that only knows English.
    - ``document_to_text`` uses per-field repetition weights instead of the
      hard-coded ``title * 8`` hack, and exposes those weights via config.
    - ``bm25_k1``/``bm25_b`` default to the conventional ``1.5/0.75`` values
      and are explicit constructor args to make grid searches straightforward.
    """

    def __init__(
        self,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        field_weights: dict[str, int] | None = None,
        add_bigrams: bool = True,
        translation_target_lang: str = "en",
    ):
        """Initialize sparse retriever parameters and caches.

        Args:
            bm25_k1: BM25+ ``k1`` saturation parameter.
            bm25_b: BM25+ ``b`` length-normalization parameter.
            field_weights: Repetition counts applied to each document field
                when serializing for indexing. Defaults to ``{"title": 3,
                "abstract": 1, "authors": 1, "venue": 1}``.
            add_bigrams: Whether to include token bigrams alongside unigrams.
            translation_target_lang: Language to translate non-English queries
                into before tokenization. Must match how the corpus was indexed.
        """
        self.bm25_model = None
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self.field_weights = dict(field_weights or DEFAULT_FIELD_WEIGHTS)
        self.add_bigrams = bool(add_bigrams)
        self.translation_target_lang = translation_target_lang
        self._indexed_corpus_fingerprint = None
        self._query_rankings_by_name = {}
        self._query_scores_by_name = {}
        self._stemmers: dict[str, object] = {"en": LancasterStemmer()}

    def _cache_key(self) -> str:
        """Build cache namespace identifier for sparse retrieval settings."""
        field_weight_tag = "-".join(
            f"{name}{weight}"
            for name, weight in sorted(self.field_weights.items())
        )
        return (
            f"bm25plus-k1_{self.bm25_k1:.2f}-b_{self.bm25_b:.2f}"
            f"-stem_pl_v2-bigrams_{int(self.add_bigrams)}"
            f"-fields_{field_weight_tag}-translate_v2"
        )

    def _get_stemmer(self, language: str):
        """Cache one stemmer per language."""
        lang = (language or "en").lower()
        if lang not in self._stemmers:
            self._stemmers[lang] = _select_stemmer(lang)
        return self._stemmers[lang]

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

        translated_query = self._translate_single_query(query, lang)
        # Keep the Lancaster (English) stemmer on post-translation query text
        # so it matches how the corpus was tokenized at index time.
        tokenized_query = self.tokenize(translated_query, language="en")
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

    def tokenize(
        self,
        text: str,
        language: str = "en",
        add_bigrams: bool | None = None,
    ) -> list[str]:
        """Tokenize text with punctuation, stopword removal, language-aware stemming."""
        if add_bigrams is None:
            add_bigrams = self.add_bigrams
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = text.split()
        stemmer = self._get_stemmer(language)
        unigrams = [stemmer.stem(t) for t in tokens if t not in STOPWORDS]
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
        corpus = [self.document_to_text(doc) for doc in collection]
        # Corpus is mostly English scientific abstracts; use the English
        # (Lancaster) tokenizer for document-side indexing.
        tokenized_corpus = [self.tokenize(text, language="en") for text in corpus]
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

        translated_queries = self._batch_translate_queries(
            queries, source_lang=lang, cache_dir=cache_dir
        )

        corpus_size = len(getattr(self.bm25_model, "doc_len", []))
        effective_top_k = corpus_size if top_k is None else min(top_k, corpus_size)
        rankings = np.empty((len(queries), effective_top_k), dtype=np.int32)
        scores = np.empty((len(queries), effective_top_k), dtype=np.float32)

        for query_idx, translated_query in enumerate(translated_queries):
            tokenized_query = self.tokenize(translated_query, language="en")
            query_scores = np.asarray(
                self.bm25_model.get_scores(tokenized_query), dtype=np.float32
            )
            ranked_indices = np.argsort(query_scores)[::-1][:effective_top_k]
            rankings[query_idx] = ranked_indices.astype(np.int32)
            scores[query_idx] = query_scores[ranked_indices].astype(np.float32)

        self._query_rankings_by_name[cache_name] = rankings
        self._query_scores_by_name[cache_name] = scores

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, rankings=rankings, scores=scores)
            print(
                f"[cache miss] Saved sparse query cache ({cache_name}) to {cache_path}"
            )

    def _translation_cache_path(self, cache_dir: str | None) -> str | None:
        """Path to the on-disk translation cache shared across query calls."""
        if cache_dir is None:
            return None
        import os

        return os.path.join(cache_dir, "sparse", "translations_v2.json")

    def _batch_translate_queries(
        self,
        queries: list[str],
        source_lang: str,
        cache_dir: str | None,
    ) -> list[str]:
        """Translate ``queries`` once, using a persistent on-disk cache when available.

        Also preserves rare original-language tokens alongside the translation
        so downstream BM25 can still match names, numbers, and proper nouns
        that survive translation poorly.
        """
        if source_lang == self.translation_target_lang:
            return list(queries)

        cache = TranslationCache(self._translation_cache_path(cache_dir))
        translated = translate_batch(
            queries,
            source_lang=source_lang,
            target_lang=self.translation_target_lang,
            cache=cache,
        )

        augmented: list[str] = []
        for original, translated_text in zip(queries, translated):
            augmented.append(
                self._augment_with_original_terms(
                    original=original, translated=translated_text
                )
            )
        return augmented

    @staticmethod
    def _augment_with_original_terms(original: str, translated: str) -> str:
        """Append long, non-stopword original tokens to the translated query."""
        seen: set[str] = set()
        kept: list[str] = []
        normalized = re.sub(r"[^\w\s]", " ", original.lower())
        for token in normalized.split():
            if token in seen or token in STOPWORDS or len(token) < 6 or not token.isalpha():
                continue
            seen.add(token)
            kept.append(token)
        if not kept:
            return translated
        return f"{translated} {' '.join(kept)}".strip()

    def _translate_single_query(self, text: str, lang: str) -> str:
        """Translate one query without using the persistent cache (for ad-hoc search)."""
        if lang == self.translation_target_lang:
            return text
        translated = translate_batch(
            [text],
            source_lang=lang,
            target_lang=self.translation_target_lang,
            cache=None,
        )
        return self._augment_with_original_terms(
            original=text, translated=translated[0]
        )

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

        Each field is repeated according to ``self.field_weights``, which
        lets BM25 approximate field-weighted scoring without maintaining
        separate per-field indices. Defaults give titles ~3x the weight of
        abstracts (down from the original 8x hack), since abstracts contain
        most of the lexical signal that links scientific tweets to papers.

        Args:
            doc (dict): dict with keys "title", "abstract", "pubkey", "authors", "venue".
                Authors is may be just names, but could also include universities, addresses, etc.
                Venue is the name of the conference/journal where the article was published. Both may be empty strings.

        Returns:
            str: A single string representation of the article.
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

        fields = {
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "venue": venue,
        }

        parts: list[str] = []
        for name, value in fields.items():
            if not value:
                continue
            weight = max(0, int(self.field_weights.get(name, 1)))
            parts.extend([value] * weight)

        return " ".join(parts).strip()

