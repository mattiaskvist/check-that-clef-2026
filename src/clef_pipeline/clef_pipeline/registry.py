from __future__ import annotations

from .pipeline import RetrievalPipeline
from .pipeline_config import PipelineConfig
from .rerankers import Gemma2BReranker, NemotronReranker
from .retrievers import BGEM3Retriever, HarrierRetriever, SparseRetriever


def create_retriever(name: str, params: dict | None = None):
    params = params or {}

    if name == "sparse":
        return SparseRetriever()
    if name == "harrier-270m":
        return HarrierRetriever(
            model_name="microsoft/harrier-oss-v1-270m",
            **params,
        )
    if name == "harrier-27b":
        return HarrierRetriever(**params)
    if name == "bge-m3":
        return BGEM3Retriever(**params)
    raise ValueError(f"Unknown retriever: {name}")


def create_reranker(name: str, params: dict | None = None):
    params = params or {}
    if name == "nemotron":
        return NemotronReranker(**params)
    if name == "gemma2b":
        return Gemma2BReranker(**params)
    raise ValueError(f"Unknown reranker: {name}")


def build_pipeline_from_config(config: PipelineConfig) -> RetrievalPipeline:
    retrievers = {
        retriever_config.name: create_retriever(
            retriever_config.name, retriever_config.params
        )
        for retriever_config in config.enabled_retrievers()
    }

    reranker = None
    if config.reranker.enabled and config.reranker.name:
        reranker = create_reranker(config.reranker.name, config.reranker.params)

    return RetrievalPipeline(config=config, retrievers=retrievers, reranker=reranker)
