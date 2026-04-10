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
        self, texts: list[str], encode_fn, cache_path: str | None, label: str
    ):
        import os

        if cache_path and os.path.exists(cache_path):
            print(f"[cache hit] Loading {label} from {cache_path}")
            embs = self.torch.load(cache_path, map_location="cuda", weights_only=True)
            print(f"Loaded {embs.shape[0]} cached {label}.")
            return embs

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

    def index(self, corpus: list[str], cache_dir: str | None = None):
        path = self._cache_path(cache_dir, "documents.pt", corpus)
        self.embeddings = self._load_or_encode(
            corpus, self.model.encode_document, path, "document embeddings"
        )

    def index_queries(
        self,
        queries: list[str],
        cache_dir: str | None = None,
        cache_name: str = "queries",
    ):
        path = self._cache_path(cache_dir, f"{cache_name}.pt", queries)
        query_embeddings = self._load_or_encode(
            queries,
            self.model.encode_query,
            path,
            f"query embeddings ({cache_name})",
        )
        self.query_embeddings = query_embeddings
        self._query_embeddings_by_name[cache_name] = query_embeddings

    def search(self, query_idx: int, cache_name: str | None = None) -> tuple[list[int], np.ndarray]:
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
        ranked = self.torch.argsort(scores, descending=True).tolist()
        scores_np = scores.cpu().numpy()
        return ranked, scores_np

    def unload_model(self):
        """Free the embedding model from GPU. Computed embeddings are kept."""
        import gc

        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()
        print("Embedding model unloaded, GPU memory freed.")


class HarrierRetriever(BaseRetriever):
    def __init__(
        self, model_name: str = "microsoft/harrier-oss-v1-0.6b", batch_size: int = 2
    ):
        import torch
        from sentence_transformers import SentenceTransformer, util

        self.util = util
        self.torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_embeddings = None
        self._query_embeddings_by_name = {}

        print(f"Loading Dense Retriever ({model_name})...")
        self.model = SentenceTransformer(
            model_name, device="cuda", model_kwargs={"dtype": "auto"}, trust_remote_code=True
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
        self, texts: list[str], cache_path: str | None, label: str, **encode_kwargs
    ):
        import os

        if cache_path and os.path.exists(cache_path):
            print(f"[cache hit] Loading {label} from {cache_path}")
            embs = self.torch.load(cache_path, map_location="cuda", weights_only=True)
            print(f"Loaded {embs.shape[0]} cached {label}.")
            return embs

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

    def index(self, corpus: list[str], cache_dir: str | None = None):
        path = self._cache_path(cache_dir, "documents.pt", corpus)
        self.embeddings = self._load_or_encode(corpus, path, "document embeddings")

    def index_queries(
        self,
        queries: list[str],
        cache_dir: str | None = None,
        cache_name: str = "queries",
    ):
        path = self._cache_path(cache_dir, f"{cache_name}.pt", queries)
        query_embeddings = self._load_or_encode(
            queries,
            path,
            f"query embeddings ({cache_name})",
            prompt_name="web_search_query",
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

    def search(self, query_idx: int, cache_name: str | None = None) -> tuple[list[int], np.ndarray]:
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
        scores_np = scores.cpu().numpy()
        return ranked, scores_np


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
        self.corpus_docs = collection
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
        except Exception as e:
            import logging
            logging.warning(f"Translation failed: {e}. Falling back to original.")
            return normalized_text or text

    def search(self, query: str, lang: str = "auto") -> tuple[list[int], np.ndarray]:
        """Translates the query if necessary, then performs BM25 search.

        Returns:
            tuple of (ranked_indices, scores) where:
                - ranked_indices: list of doc indices sorted by BM25 score descending
                - scores: numpy array where scores[doc_id] = BM25 score
        """
        translated_query = self._translate_query(query, lang)
        tokenized_query = self.tokenize(translated_query)
        scores = self.bm25_model.get_scores(tokenized_query)
        ranked = np.argsort(scores)[::-1].tolist()
        return ranked, scores

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


# ==========================================
# HARD INDICATOR RETRIEVER INTERNALS
# ==========================================

class HardIndicatorRetriever:
    """Natively computes hard indicator scores for candidates."""
    
    _COVID_BLOCKLIST = {
        "covid", "covid-19", "covid19", "covid 19",
        "sars-cov-2", "sars-cov2", "sarscov2", "sars cov 2",
        "coronavirus", "corona", "corona virus",
        "pandemic", "the pandemic",
    }
    
    def __init__(self):
        self._nlp_model = None
        self._cc_instance = None
        
        self.extractors = {
            "persons":     (self._extract_persons,     "fuzzy"),
            "locations":   (self._extract_locations,   "location"),
            "orgs":        (self._extract_orgs,        "fuzzy"),
        }

    def _filter_covid(self, entities: list[str]) -> list[str]:
        return [e for e in entities if e.lower().strip() not in self._COVID_BLOCKLIST]

    def _get_nlp(self):
        if self._nlp_model is None:
            import sys
            _torch_backup = sys.modules.get("torch", "__MISSING__")
            sys.modules["torch"] = None
            try:
                import spacy
                self._nlp_model = spacy.load("en_core_web_sm")
            finally:
                if _torch_backup == "__MISSING__":
                    sys.modules.pop("torch", None)
                else:
                    sys.modules["torch"] = _torch_backup
        return self._nlp_model

    def _extract_persons(self, text: str) -> list[str]:
        return self._filter_covid([ent.text for ent in self._get_nlp()(text).ents if ent.label_ == "PERSON"])

    def _extract_locations(self, text: str) -> list[str]:
        return self._filter_covid([ent.text for ent in self._get_nlp()(text).ents if ent.label_ in ("GPE", "LOC")])

    def _extract_orgs(self, text: str) -> list[str]:
        return self._filter_covid([ent.text for ent in self._get_nlp()(text).ents if ent.label_ == "ORG"])

    def _get_cc(self):
        if self._cc_instance is None:
            import logging
            logging.getLogger("country_converter").setLevel(logging.ERROR)
            import country_converter as coco
            self._cc_instance = coco.CountryConverter()
        return self._cc_instance
        
    def _normalize_location(self, entity: str) -> list[str]:
        from geotext import GeoText
        terms = [entity.lower().strip()]
        country = self._get_cc().convert(names=entity, to="name_short", not_found=None)
        if country:
            terms.append(country.lower().strip())
        geo = GeoText(entity.title())
        if geo.country_mentions:
            iso = list(geo.country_mentions.keys())[0]
            parent = self._get_cc().convert(names=iso, to="name_short", not_found=None)
            if parent:
                terms.append(parent.lower().strip())
        return list(set(terms))

    def _matches(self, entities: list[str], target: str, strategy: str, threshold: int = 80) -> bool:
        from rapidfuzz import fuzz
        if not entities or not target:
            return False
        target_lower = target.lower()
        for entity in entities:
            if strategy == "exact":
                if entity.lower().strip() in target_lower:
                    return True
            elif strategy == "location":
                for term in self._normalize_location(entity):
                    if fuzz.token_set_ratio(term, target_lower) >= threshold:
                        return True
            else:
                if fuzz.token_set_ratio(entity.lower().strip(), target_lower) >= threshold:
                    return True
        return False
        
    def _score_single(self, query: str, document: str) -> float:
        total_score = 0.0
        for extractor_fn, strategy in self.extractors.values():
            entities = extractor_fn(query)
            for entity in entities:
                if self._matches([entity], document, strategy):
                    total_score += 1.0
                else:
                    total_score -= 1.0
        return total_score

    def score_candidates(self, query: str, candidates: list[int], article_texts: list[str]) -> dict[int, float]:
        """Scores a list of document candidates for a given query.
        
        Args:
            query (str): The search query text.
            candidates (list[int]): List of document indices to score.
            article_texts (list[str]): Complete list of article texts where index matches doc_id.
            
        Returns:
            dict[int, float]: A dictionary mapping doc_id to its hard indicator score.
        """
        indicator_scores = {}
        for doc_id in candidates:
            article_text = article_texts[doc_id]
            indicator_scores[doc_id] = self._score_single(query, article_text)
        return indicator_scores
