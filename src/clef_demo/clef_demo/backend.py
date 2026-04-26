import modal

app = modal.App("clef-backend")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "datasets",
        "transformers",
        "accelerate",
        "peft",
        "sentence-transformers",
        "rank_bm25",
        "tqdm",
        "nltk",
        "Pillow",
        "torchvision",
        "deep-translator",
    )
    .add_local_python_source("clef_demo", "clef_pipeline")
)

embedding_cache = modal.Volume.from_name(
    "checkthat-embedding-cache", create_if_missing=True
)


@app.cls(
    image=image,
    gpu="A10G", # swap to L40S for real demo for qwen
    timeout=int(
        60 * 60 * 0.5
    ),  # 15 mins max runtime to avoid unexpected long-running costs
    volumes={"/cache/embeddings": embedding_cache},
    secrets=[modal.Secret.from_name("hf-token")],
    scaledown_window=150,  # Keeps GPU alive for 2.5 mins
)
class PipelineBackend:
    @modal.method()
    def load_cached_collection(self, selected_retrievers: list[str]):
        from clef_pipeline.pipeline_config import (
            PipelineConfig,
            RerankerConfig,
            RetrieverConfig,
        )
        from clef_pipeline.registry import build_pipeline_from_config
        from clef_pipeline.utils import CHECKTHAT_DATASET
        from datasets import load_dataset

        config = PipelineConfig(
            retrievers=[RetrieverConfig(name=name) for name in selected_retrievers],
            reranker=RerankerConfig(name=None, enabled=False),
            use_fusion=True,
            fusion_top_k=30,
            sparse_cache_top_k=2000,
            final_top_k=5,
        )

        base_docs = load_dataset(
            CHECKTHAT_DATASET, "collection", split="collection"
        ).to_list()
        self.pipeline = build_pipeline_from_config(config)
        self.pipeline.index_collection(base_docs, cache_dir="/cache/embeddings")

        # only grab the first 5 documents to send back as a preview
        # The full dataset stays safely in the GPU's memory,
        # while keeping the response lightweight and fast for the UI
        preview_docs = self.pipeline.collection_documents[:5]

        indexed_docs_summary = [
            {
                "pubkey": doc.get("pubkey"),
                "title": doc.get("title"),
                "abstract": doc.get("abstract"),
                "venue": doc.get("venue"),
                "authors": doc.get("authors"),
            }
            for doc in preview_docs
        ]
        return len(self.pipeline.collection_documents), indexed_docs_summary

    @modal.method()
    def search(
        self,
        query_text: str,
        enable_fusion: bool = True,
        fusion_method: str = "rrf",
        fusion_top_k: int = 30,
        reranker_name: str = "none",
    ):
        if not hasattr(self, "pipeline") or self.pipeline is None:
            raise RuntimeError(
                "Pipeline not initialized. The container may have restarted. Load cached embeddings first."
            )

        from clef_pipeline.pipeline import RetrievalPipeline
        from clef_pipeline.pipeline_config import PipelineConfig, RerankerConfig
        from clef_pipeline.registry import create_reranker

        reranker_name = (reranker_name or "none").strip().lower()
        if reranker_name == "none":
            runtime_reranker = None
            runtime_reranker_config = RerankerConfig(name=None, enabled=False)
        else:
            runtime_reranker = create_reranker(reranker_name)
            runtime_reranker_config = RerankerConfig(name=reranker_name, enabled=True)

        runtime_config = PipelineConfig(
            retrievers=self.pipeline.config.retrievers,
            reranker=runtime_reranker_config,
            use_fusion=bool(enable_fusion),
            fusion_method=fusion_method,
            fusion_top_k=int(fusion_top_k),
            sparse_cache_top_k=self.pipeline.config.sparse_cache_top_k,
            final_top_k=self.pipeline.config.final_top_k,
            hf_fusion_repo_id=self.pipeline.config.hf_fusion_repo_id,
            hf_token=self.pipeline.config.hf_token,
        )

        runtime_pipeline = RetrievalPipeline.clone_runtime(
            base=self.pipeline,
            config=runtime_config,
            reranker=runtime_reranker,
        )

        result = runtime_pipeline.search_text(query_text, lang="auto")
        preds = result["preds"]
        docs_by_pubkey = {
            doc["pubkey"]: doc for doc in runtime_pipeline.collection_documents
        }

        rows = []
        for rank, pubkey in enumerate(preds, start=1):
            doc = docs_by_pubkey.get(pubkey, {})
            rows.append(
                {
                    "rank": rank,
                    "pubkey": pubkey,
                    "title": doc.get("title", ""),
                    "abstract": doc.get("abstract", ""),
                    "authors": doc.get("authors", ""),
                }
            )

        return {"rows": rows, "stages": result["stages"]}
