"""Core retrieval pipeline orchestration and stage execution logic."""

from __future__ import annotations

import uuid
from collections import OrderedDict

from .fusions import RRFFuser, RandomForestFuser
from .interfaces import BaseReranker
from .pipeline_config import PipelineConfig
from .utils import translate_texts


class RetrievalPipeline:
    """Coordinate indexing, retrieval, fusion, and reranking stages."""

    DEFAULT_DENSE_DOC_MAX_CHARS = 2048

    def __init__(
        self,
        config: PipelineConfig,
        retrievers: dict[str, object],
        reranker: object | None,
    ):
        """Create a pipeline instance from configured components.

        Args:
            config: Pipeline behavior and component settings.
            retrievers: Retriever instances keyed by retriever name.
            reranker: Optional reranker instance.

        Raises:
            KeyError: If a configured retriever implementation is missing.
        """
        self.config = config
        self.retrievers = OrderedDict()
        for retriever_config in self.config.enabled_retrievers():
            name = retriever_config.name
            if name not in retrievers:
                raise KeyError(f"Missing retriever implementation for '{name}'.")
            self.retrievers[name] = retrievers[name]

        self.reranker = reranker if self.config.reranker.enabled else None
        if self.config.fusion_method == "rrf":
            self.fuser = RRFFuser()
        elif self.config.fusion_method == "random_forest":
            self.fuser = RandomForestFuser(
                hf_repo_id=self.config.hf_fusion_repo_id,
                hf_token=self.config.hf_token,
            )
        else:
            raise ValueError(
                f"Unknown fusion method: {self.config.fusion_method}. "
                "Use 'rrf' or 'random_forest'."
            )
        self.collection_documents: list[dict] = []
        self.article_pubkeys: list[str] = []
        self.reranker_corpus: list[str] = []
        self._cache_names: dict[tuple[str, str], str] = {}

    @staticmethod
    def _document_to_text(doc: dict) -> str:
        """Normalize a document dictionary into retrieval-ready text."""
        return BaseReranker.document_to_text(doc)

    @staticmethod
    def _merge_documents(
        base_documents: list[dict], custom_documents: list[dict] | None
    ) -> list[dict]:
        """Merge custom documents into the base collection by ``pubkey``.

        Args:
            base_documents: Dataset documents.
            custom_documents: User-provided documents overriding by key.

        Returns:
            Merged document list preserving base order unless overridden.
        """
        if not custom_documents:
            return list(base_documents)

        by_pubkey = OrderedDict(
            (str(doc["pubkey"]), dict(doc)) for doc in base_documents
        )
        for custom_doc in custom_documents:
            pubkey = str(custom_doc["pubkey"])
            by_pubkey[pubkey] = dict(custom_doc)
        return list(by_pubkey.values())

    def _index_one_retriever(
        self,
        name: str,
        retriever: object,
        article_texts: list[str],
        force_recompute_dense_documents: bool = False,
        cache_dir: str | None = None,
    ):
        """Index collection documents for one retriever.

        Args:
            name: Retriever registry name.
            retriever: Retriever instance.
            article_texts: Textified collection documents.
            force_recompute_dense_documents: Whether dense caches are bypassed.
            cache_dir: Cache root directory for embedders.
        """
        if name.startswith("sparse"):
            retriever.index(self.collection_documents)
            return

        try:
            retriever.index(
                article_texts,
                cache_dir=cache_dir,
                force_recompute=force_recompute_dense_documents,
            )
        except TypeError:
            retriever.index(article_texts)

    def _index_query_cache(
        self,
        retriever_name: str,
        retriever: object,
        cache_name: str,
        query_texts: list[str],
        lang: str,
        cache_dir: str | None,
        force_recompute_sparse_cache: bool,
        force_recompute_dense_queries: bool,
    ):
        """Build or load cached query representations for one retriever.

        Args:
            retriever_name: Retriever registry name.
            retriever: Retriever instance.
            cache_name: Retriever-specific query cache identifier.
            query_texts: Queries to index.
            lang: Query language for sparse translation/tokenization path.
            cache_dir: Cache root directory.
            force_recompute_sparse_cache: Whether sparse query cache is bypassed.
            force_recompute_dense_queries: Whether dense query cache is bypassed.
        """
        if retriever_name.startswith("sparse"):
            try:
                retriever.index_queries(
                    query_texts,
                    lang=lang,
                    cache_dir=cache_dir,
                    cache_name=cache_name,
                    top_k=self.config.sparse_cache_top_k,
                    force_recompute=force_recompute_sparse_cache,
                )
            except TypeError:
                retriever.index_queries(query_texts)
            return

        try:
            retriever.index_queries(
                query_texts,
                cache_dir=cache_dir,
                cache_name=cache_name,
                force_recompute=force_recompute_dense_queries,
            )
        except TypeError:
            retriever.index_queries(query_texts)

    def _dense_doc_text_limit(self) -> int:
        """Resolve the max character budget for dense document indexing."""
        return self.DEFAULT_DENSE_DOC_MAX_CHARS

    @classmethod
    def clone_runtime(
        cls,
        base: "RetrievalPipeline",
        config: PipelineConfig,
        reranker: object | None,
    ) -> "RetrievalPipeline":
        """Create a runtime-tuned pipeline without re-indexing retrievers.

        This is used to swap inference-only settings (fusion method, reranker)
        while reusing the already-loaded document collection and retriever state.
        """
        pipeline = cls(
            config=config, retrievers=dict(base.retrievers), reranker=reranker
        )
        pipeline.collection_documents = base.collection_documents
        pipeline.article_pubkeys = base.article_pubkeys
        if pipeline.reranker is not None:
            pipeline.reranker_corpus = pipeline.reranker.preprocess_corpus(
                pipeline.collection_documents
            )
        else:
            pipeline.reranker_corpus = base.reranker_corpus
        pipeline._cache_names = dict(base._cache_names)
        return pipeline

    def index_collection(
        self,
        collection_documents: list[dict],
        custom_documents: list[dict] | None = None,
        cache_dir: str | None = None,
        force_recompute_dense_documents: bool = False,
    ):
        """Index the document collection for all configured retrievers.

        Args:
            collection_documents: Base collection records.
            custom_documents: Optional user additions or overrides by ``pubkey``.
            cache_dir: Cache root path for dense embedding artifacts.
            force_recompute_dense_documents: Whether dense doc cache is bypassed.
        """
        self.collection_documents = self._merge_documents(
            collection_documents, custom_documents
        )
        self.article_pubkeys = [doc["pubkey"] for doc in self.collection_documents]
        dense_doc_text_limit = self._dense_doc_text_limit()
        article_texts = [
            self._document_to_text(doc)[:dense_doc_text_limit]
            for doc in self.collection_documents
        ]

        if self.reranker is not None:
            self.reranker_corpus = self.reranker.preprocess_corpus(
                self.collection_documents
            )
        else:
            self.reranker_corpus = article_texts

        for retriever_name, retriever in self.retrievers.items():
            self._index_one_retriever(
                retriever_name,
                retriever,
                article_texts=article_texts,
                force_recompute_dense_documents=force_recompute_dense_documents,
                cache_dir=cache_dir,
            )

    def index_queries_for_language(
        self,
        lang: str,
        query_texts: list[str],
        cache_lang: str | None = None,
        cache_dir: str | None = None,
        force_recompute_sparse_cache: bool = False,
        force_recompute_dense_queries: bool = False,
    ):
        """Index query representations for each retriever under one language key.

        Args:
            lang: Query language code used by retrievers.
            query_texts: Query strings for this language.
            cache_lang: Optional cache namespace override.
            cache_dir: Cache root path for query artifacts.
            force_recompute_sparse_cache: Whether sparse query cache is bypassed.
            force_recompute_dense_queries: Whether dense query cache is bypassed.
        """
        cache_lang_key = cache_lang or lang
        for retriever_name, retriever in self.retrievers.items():
            cache_name = f"{retriever_name}_queries_{cache_lang_key}"
            self._cache_names[(retriever_name, cache_lang_key)] = cache_name
            self._index_query_cache(
                retriever_name=retriever_name,
                retriever=retriever,
                cache_name=cache_name,
                query_texts=query_texts,
                lang=lang,
                cache_dir=cache_dir,
                force_recompute_sparse_cache=force_recompute_sparse_cache,
                force_recompute_dense_queries=force_recompute_dense_queries,
            )

    def _stage_pubkeys(self, ranked_indices: list[int]) -> list[str]:
        """Map ranked document indices to publication keys."""
        return [
            self.article_pubkeys[idx]
            for idx in ranked_indices
            if 0 <= idx < len(self.article_pubkeys)
        ]

    @staticmethod
    def _resolve_dense_sparse_stage_keys(
        ranked_indices_by_stage: dict[str, list[int]],
    ) -> tuple[str | None, str | None]:
        """Resolve dense and sparse stage keys from retriever-stage outputs."""
        dense_key = next(
            (key for key in ranked_indices_by_stage if not key.startswith("sparse")),
            None,
        )
        sparse_key = next(
            (key for key in ranked_indices_by_stage if key.startswith("sparse")),
            None,
        )
        return dense_key, sparse_key

    def _run_retrievers_for_query(
        self, query_idx: int, cache_lang: str
    ) -> tuple[dict[str, list[int]], dict[str, list[float]]]:
        """Run all retrievers for one indexed query.

        Args:
            query_idx: Query position in indexed query cache.
            cache_lang: Cache language key used when indexing queries.

        Returns:
            Ranked indices per retriever stage and optional stage score lists.

        Raises:
            KeyError: If a retriever cache was not indexed for the language.
        """
        ranked_indices_by_stage: dict[str, list[int]] = {}
        score_lists_by_stage: dict[str, list[float]] = {}
        for retriever_name, retriever in self.retrievers.items():
            cache_name = self._cache_names.get((retriever_name, cache_lang))
            if cache_name is None:
                raise KeyError(
                    f"No indexed query cache for retriever '{retriever_name}' and lang '{cache_lang}'."
                )

            search_with_scores = getattr(retriever, "search_with_scores", None)
            if self.config.fusion_method == "random_forest" and callable(
                search_with_scores
            ):
                ranked_indices, scores = search_with_scores(
                    query_idx, cache_name=cache_name
                )
                ranked_indices_by_stage[retriever_name] = ranked_indices
                score_lists_by_stage[retriever_name] = scores
                continue

            ranked_indices_by_stage[retriever_name] = retriever.search(
                query_idx, cache_name=cache_name
            )
        return ranked_indices_by_stage, score_lists_by_stage

    def _choose_candidates(
        self,
        ranked_indices_by_stage: dict[str, list[int]],
        score_lists_by_stage: dict[str, list[float]],
        lang: str,
    ) -> list[int]:
        """Select candidate document indices from retriever outputs."""
        if not ranked_indices_by_stage:
            return []
        if self.config.use_fusion and len(ranked_indices_by_stage) > 1:
            if self.config.fusion_method == "random_forest":
                dense_key, sparse_key = self._resolve_dense_sparse_stage_keys(
                    ranked_indices_by_stage
                )
                if dense_key is None or sparse_key is None:
                    raise ValueError(
                        "Random forest fusion requires one dense and one sparse retriever."
                    )
                dense_scores = score_lists_by_stage.get(dense_key)
                sparse_scores = score_lists_by_stage.get(sparse_key)
                if dense_scores is None or sparse_scores is None:
                    raise ValueError(
                        "Random forest fusion requires score lists from both retrievers."
                    )
                return self.fuser.fuse(
                    ranked_lists=[
                        ranked_indices_by_stage[dense_key],
                        ranked_indices_by_stage[sparse_key],
                    ],
                    scores_lists=[dense_scores, sparse_scores],
                    top_k=self.config.fusion_top_k,
                    lang=lang,
                )
            return self.fuser.fuse(
                ranked_lists=list(ranked_indices_by_stage.values()),
                top_k=self.config.fusion_top_k,
            )
        first_stage = next(iter(ranked_indices_by_stage.values()))
        return first_stage[: self.config.fusion_top_k]

    def _apply_reranker(
        self, query_text: str, candidate_indices: list[int]
    ) -> list[int]:
        """Apply reranking over fusion candidates when reranker is enabled."""
        if self.reranker is None or not candidate_indices:
            return candidate_indices
        corpus = self.reranker_corpus
        target_language = self.config.target_language
        if target_language:
            corpus = list(corpus)
            per_lang: dict[str, list[tuple[int, str]]] = {}
            for idx in candidate_indices:
                if 0 <= idx < len(self.collection_documents):
                    doc = self.collection_documents[idx]
                    doc_lang = str(doc.get("lang") or doc.get("language") or "auto")
                else:
                    doc_lang = "auto"
                per_lang.setdefault(doc_lang, []).append((idx, corpus[idx]))

            for doc_lang, items in per_lang.items():
                translated = translate_texts(
                    [text for _idx, text in items],
                    source_language=doc_lang,
                    target_language=target_language,
                )
                for (doc_idx, _text), translated_text in zip(items, translated):
                    corpus[doc_idx] = translated_text
        reranked = self.reranker.rerank(
            query=query_text,
            doc_indices=candidate_indices,
            corpus=corpus,
        )
        return [doc_idx for doc_idx, _score in reranked]

    def search_cached_query(
        self,
        query_idx: int,
        query_text: str,
        lang: str,
        cache_lang: str | None = None,
    ) -> dict[str, object]:
        """Search using pre-indexed query caches and return stage outputs.

        Args:
            query_idx: Query position in language cache.
            query_text: Original query text for reranker input.
            lang: Actual query language code.
            cache_lang: Optional cache key namespace. Defaults to ``lang``.

        Returns:
            Dict containing final predictions and per-stage publication keys.
        """
        ranked_indices_by_stage, score_lists_by_stage = self._run_retrievers_for_query(
            query_idx=query_idx, cache_lang=cache_lang or lang
        )
        candidate_indices = self._choose_candidates(
            ranked_indices_by_stage=ranked_indices_by_stage,
            score_lists_by_stage=score_lists_by_stage,
            lang=lang,
        )
        final_indices = self._apply_reranker(query_text, candidate_indices)

        dense_key, sparse_key = self._resolve_dense_sparse_stage_keys(
            ranked_indices_by_stage
        )
        stages = {
            "dense": self._stage_pubkeys(ranked_indices_by_stage.get(dense_key, [])),
            "sparse": self._stage_pubkeys(ranked_indices_by_stage.get(sparse_key, [])),
            "rrf": self._stage_pubkeys(candidate_indices),
            "final": self._stage_pubkeys(final_indices),
        }
        return {
            "preds": stages["final"][: self.config.final_top_k],
            "stages": stages,
        }

    def search_text(self, query_text: str, lang: str = "en") -> dict[str, object]:
        """Search raw query text by creating an ad-hoc query cache entry.

        Args:
            query_text: Raw query string.
            lang: Query language code.

        Returns:
            Final predictions and stage outputs for the query.
        """
        cache_lang = f"{lang}_adhoc_{uuid.uuid4().hex[:8]}"
        self.index_queries_for_language(
            lang=lang,
            query_texts=[query_text],
            cache_lang=cache_lang,
        )
        return self.search_cached_query(
            query_idx=0,
            query_text=query_text,
            lang=lang,
            cache_lang=cache_lang,
        )

    def get_fusion_retrievers(self) -> tuple[object, object]:
        """Return dense and sparse retriever instances used by fusion."""
        dense_name = next(
            (name for name in self.retrievers.keys() if not name.startswith("sparse")),
            None,
        )
        sparse_name = next(
            (name for name in self.retrievers.keys() if name.startswith("sparse")),
            None,
        )
        if dense_name is None or sparse_name is None:
            raise ValueError(
                "Pipeline must have one dense and one sparse retriever for fusion."
            )
        return self.retrievers[dense_name], self.retrievers[sparse_name]

    def unload_dense_models(self):
        """Unload dense retriever models from GPU while keeping cached embeddings."""
        for retriever_name, retriever in self.retrievers.items():
            if retriever_name.startswith("sparse"):
                continue
            unload_model = getattr(retriever, "unload_model", None)
            if callable(unload_model):
                unload_model()

    def prepare_fusion_model(
        self,
        cache_dir: str,
        languages: list[str],
        force_retrain_fusion: bool = False,
        force_recompute_dense_queries: bool = False,
        force_recompute_sparse_cache: bool = False,
        on_cache_update=None,
    ):
        """Load or train the fusion model for all configured languages.

        Args:
            cache_dir: Cache root path.
            languages: List of language codes to train for.
            force_retrain_fusion: Force retraining instead of loading cache.
            force_recompute_dense_queries: Force recompute dense train queries.
            force_recompute_sparse_cache: Force recompute sparse train queries.
            on_cache_update: Callback for when the embedding cache changes.
        """
        if self.config.fusion_method != "random_forest":
            return

        from datasets import load_dataset
        from .utils import CHECKTHAT_DATASET

        dense_retriever, sparse_retriever = self.get_fusion_retrievers()
        if not isinstance(self.fuser, RandomForestFuser):
            raise RuntimeError("Expected RandomForestFuser for random_forest mode.")

        sparse_config = {
            "k1": getattr(sparse_retriever, "bm25_k1", None),
            "b": getattr(sparse_retriever, "bm25_b", None),
            "stemmer": "lancaster",
        }
        dense_model_name = getattr(
            dense_retriever, "model_name", dense_retriever.__class__.__name__
        )

        loaded = False
        if not force_retrain_fusion:
            loaded = self.fuser.load(
                cache_dir=cache_dir,
                dense_model_name=dense_model_name,
                sparse_config=sparse_config,
                train_split="train",
            )

        if not loaded:
            train_tweets_by_lang: dict[str, list[dict]] = {}
            for lang in languages:
                train_tweets = list(load_dataset(CHECKTHAT_DATASET, lang)["train"])
                train_tweets_by_lang[lang] = train_tweets
                train_query_texts = [row["text"] for row in train_tweets]

                dense_retriever.index_queries(
                    train_query_texts,
                    cache_dir=cache_dir,
                    cache_name=f"queries_train_{lang}",
                    force_recompute=force_recompute_dense_queries,
                )
                sparse_retriever.index_queries(
                    train_query_texts,
                    lang=lang,
                    cache_dir=cache_dir,
                    cache_name=f"sparse_queries_train_{lang}",
                    top_k=self.config.sparse_cache_top_k,
                    force_recompute=force_recompute_sparse_cache,
                )
                if on_cache_update is not None:
                    on_cache_update()

            self.fuser.train(
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
                train_tweets_by_lang=train_tweets_by_lang,
                article_pubkeys=self.article_pubkeys,
                dense_model_name=dense_model_name,
                sparse_config=sparse_config,
                train_split="train",
            )
            self.fuser.save(cache_dir=cache_dir)
            if on_cache_update is not None:
                on_cache_update()
