from __future__ import annotations

import uuid
from collections import OrderedDict

from .interfaces import BaseReranker
from .pipeline_config import PipelineConfig
from .utils import FusionProcessor


class RetrievalPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        retrievers: dict[str, object],
        reranker: object | None,
    ):
        self.config = config
        self.retrievers = OrderedDict()
        for retriever_config in self.config.enabled_retrievers():
            name = retriever_config.name
            if name not in retrievers:
                raise KeyError(f"Missing retriever implementation for '{name}'.")
            self.retrievers[name] = retrievers[name]

        self.reranker = reranker if self.config.reranker.enabled else None
        self.fusion = FusionProcessor()
        self.collection_documents: list[dict] = []
        self.article_pubkeys: list[str] = []
        self.reranker_corpus: list[str] = []
        self._cache_names: dict[tuple[str, str], str] = {}

    @staticmethod
    def _document_to_text(doc: dict) -> str:
        return BaseReranker.document_to_text(doc)

    @staticmethod
    def _merge_documents(
        base_documents: list[dict], custom_documents: list[dict] | None
    ) -> list[dict]:
        if not custom_documents:
            return list(base_documents)

        by_pubkey = OrderedDict((str(doc["pubkey"]), dict(doc)) for doc in base_documents)
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

    def index_collection(
        self,
        collection_documents: list[dict],
        custom_documents: list[dict] | None = None,
        cache_dir: str | None = None,
        force_recompute_dense_documents: bool = False,
    ):
        self.collection_documents = self._merge_documents(collection_documents, custom_documents)
        self.article_pubkeys = [doc["pubkey"] for doc in self.collection_documents]
        article_texts = [self._document_to_text(doc) for doc in self.collection_documents]

        if self.reranker is not None:
            self.reranker_corpus = self.reranker.preprocess_corpus(self.collection_documents)
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
        return [
            self.article_pubkeys[idx]
            for idx in ranked_indices
            if 0 <= idx < len(self.article_pubkeys)
        ]

    def _run_retrievers_for_query(self, query_idx: int, lang: str) -> dict[str, list[int]]:
        ranked_indices_by_stage: dict[str, list[int]] = {}
        for retriever_name, retriever in self.retrievers.items():
            cache_name = self._cache_names.get((retriever_name, lang))
            if cache_name is None:
                raise KeyError(f"No indexed query cache for retriever '{retriever_name}' and lang '{lang}'.")
            ranked_indices_by_stage[retriever_name] = retriever.search(
                query_idx, cache_name=cache_name
            )
        return ranked_indices_by_stage

    def _choose_candidates(self, ranked_indices_by_stage: dict[str, list[int]]) -> list[int]:
        if not ranked_indices_by_stage:
            return []
        if self.config.use_fusion and len(ranked_indices_by_stage) > 1:
            return self.fusion.reciprocal_rank_fusion(
                list(ranked_indices_by_stage.values()),
                top_k=self.config.fusion_top_k,
            )
        first_stage = next(iter(ranked_indices_by_stage.values()))
        return first_stage[: self.config.fusion_top_k]

    def _apply_reranker(self, query_text: str, candidate_indices: list[int]) -> list[int]:
        if self.reranker is None or not candidate_indices:
            return candidate_indices
        reranked = self.reranker.rerank(
            query=query_text,
            doc_indices=candidate_indices,
            corpus=self.reranker_corpus,
        )
        return [doc_idx for doc_idx, _score in reranked]

    def search_cached_query(self, query_idx: int, query_text: str, lang: str) -> dict[str, object]:
        ranked_indices_by_stage = self._run_retrievers_for_query(query_idx=query_idx, lang=lang)
        candidate_indices = self._choose_candidates(ranked_indices_by_stage)
        final_indices = self._apply_reranker(query_text, candidate_indices)

        dense_key = next((key for key in ranked_indices_by_stage if key.startswith("harrier") or key.startswith("bge")), None)
        sparse_key = next((key for key in ranked_indices_by_stage if key.startswith("sparse")), None)
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
        cache_lang = f"{lang}_adhoc_{uuid.uuid4().hex[:8]}"
        self.index_queries_for_language(
            lang=lang,
            query_texts=[query_text],
            cache_lang=cache_lang,
        )
        return self.search_cached_query(query_idx=0, query_text=query_text, lang=cache_lang)

    def unload_dense_models(self):
        for retriever_name, retriever in self.retrievers.items():
            if retriever_name.startswith("sparse"):
                continue
            unload_model = getattr(retriever, "unload_model", None)
            if callable(unload_model):
                unload_model()
