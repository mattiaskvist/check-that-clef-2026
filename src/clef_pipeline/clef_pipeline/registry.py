"""Factory helpers for constructing retrievers, rerankers, and pipelines."""

from __future__ import annotations

from .pipeline import RetrievalPipeline
from .pipeline_config import PipelineConfig
from .rerankers import Gemma2BReranker, JinaReranker, NemotronReranker, Qwen3Reranker
from .retrievers import BGEM3Retriever, HarrierRetriever, SparseRetriever

BGE_M3_LORA_ID = "boyes-boys-clef-2026/bge-m3-checkthat-finetuned"


def create_retriever(name: str, params: dict | None = None):
    """Instantiate a retriever by registry name.

    Args:
        name: Retriever identifier.
        params: Optional constructor keyword arguments.

    Returns:
        Concrete retriever instance.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    params = params or {}

    if name == "sparse":
        return SparseRetriever(**params)
    if name == "harrier-270m":
        return HarrierRetriever(
            model_name="microsoft/harrier-oss-v1-270m",
            **params,
        )
    if name == "harrier-27b":
        return HarrierRetriever(**params)
    if name == "bge-m3":
        bge_params = {"lora_id": BGE_M3_LORA_ID, **params}
        return BGEM3Retriever(**bge_params)
    raise ValueError(f"Unknown retriever: {name}")


def create_reranker(name: str, params: dict | None = None):
    """Instantiate a reranker by registry name.

    Args:
        name: Reranker identifier.
        params: Optional constructor keyword arguments.

    Returns:
        Concrete reranker instance.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    params = params or {}
    if name == "nemotron":
        return NemotronReranker(**params)
    if name == "gemma2b":
        return Gemma2BReranker(**params)
    if name == "jina-v3":
        return JinaReranker(**params)
    if name == "qwen3-reranker-8b":
        return Qwen3Reranker(model_name="Qwen/Qwen3-Reranker-8B", **params)
    if name == "qwen3-reranker-4b":
        return Qwen3Reranker(model_name="Qwen/Qwen3-Reranker-4B", **params)
    if name == "qwen3-reranker-0.6b":
        return Qwen3Reranker(model_name="Qwen/Qwen3-Reranker-0.6B", **params)
    raise ValueError(f"Unknown reranker: {name}")


def build_pipeline_from_config(config: PipelineConfig) -> RetrievalPipeline:
    """Build a retrieval pipeline and all enabled components from config.

    Args:
        config: Pipeline settings and component declarations.

    Returns:
        Fully constructed retrieval pipeline.
    """
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
