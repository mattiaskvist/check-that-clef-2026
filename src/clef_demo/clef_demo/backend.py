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
    gpu="A10G", # swap to A100-80GBS for real demo for qwen
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
            fusion_method="random_forest",
            hf_fusion_repo_id="boyes-boys-clef-2026/random-forest-fuser",
            fusion_top_k=30,
            sparse_cache_top_k=2000,
            final_top_k=5,
        )

        base_docs = load_dataset(
            CHECKTHAT_DATASET, "collection", split="collection"
        ).to_list()
        self.pipeline = build_pipeline_from_config(config)
        self.pipeline.index_collection(base_docs, cache_dir="/cache/embeddings")

        import os
        import joblib
        
        fixed_path = "/cache/embeddings/hf_hub/models--boyes-boys-clef-2026--random-forest-fuser/snapshots/db30a2f44670a5bc6d81ff7f8c0e1b8719de20d9/rf_1d2c99cd554bfc4b.pkl"

        if os.path.exists(fixed_path):
            print(f"Force-loading existing model: {fixed_path}")
            payload = joblib.load(fixed_path)
            
            if isinstance(payload, dict) and "model" in payload:
                self.pipeline.fuser.model = payload["model"]
            else:
                self.pipeline.fuser.model = payload # Fallback if it was just the model
                
            self.pipeline.fuser._trained = True
        else:
            print("Fixed path not found, running standard prepare...")
            self.pipeline.prepare_fusion_model(
                cache_dir="/cache/embeddings",
                languages=["en", "fr", "de"],
                force_retrain_fusion=False
            )

        # only grab the first 5 documents to send back as a preview
        # The full dataset stays safely in the GPU's memory,
        # while keeping the response lightweight and fast for the UI
        preview_docs = self.pipeline.collection_documents[:15]

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
        fusion_method: str = "random_forest",
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

        if fusion_method == "none":
            enable_fusion = False
            fusion_method = "rrf"  # dummy fallback

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

        if runtime_config.fusion_method == "random_forest":
            from clef_pipeline.fusions import RandomForestFuser
            if isinstance(self.pipeline.fuser, RandomForestFuser) and self.pipeline.fuser.model:
                runtime_pipeline.fuser.model = self.pipeline.fuser.model
                runtime_pipeline.fuser._trained = True
            else:
                # If we have to load from disk in the search call
                import joblib, os
                fixed_path = "..." 
                if os.path.exists(fixed_path):
                    payload = joblib.load(fixed_path)
                    runtime_pipeline.fuser.model = payload["model"] if isinstance(payload, dict) else payload
                    runtime_pipeline.fuser._trained = True

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
