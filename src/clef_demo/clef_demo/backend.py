"""Modal GPU backend used by the Streamlit demo."""

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
    gpu="A10G",
    timeout=int(
        60 * 60 * 0.5
    ),  # 15 mins max runtime to avoid unexpected long-running costs
    volumes={"/cache/embeddings": embedding_cache},
    secrets=[modal.Secret.from_name("hf-token")],
    scaledown_window=150,  # Keeps GPU alive for 2.5 mins
)
class PipelineBackend:
    """Stateful Modal class that owns one in-memory retrieval pipeline.

    Streamlit first calls ``index_documents`` to load data and build retriever
    indexes inside the GPU container. Later ``search`` calls reuse that same
    in-memory pipeline as long as the Modal container remains alive.
    """

    @modal.method()
    def index_documents(
        self,
        selected_retrievers: list[str],
        enable_fusion: bool,
        enable_reranker: bool,
        fusion_top_k: int,
        custom_docs: list[dict],
    ):
        """Load the collection and build a demo pipeline in the Modal container.

        Args:
            selected_retrievers: Registry names selected in the UI.
            enable_fusion: Whether multiple retriever outputs should be fused.
            enable_reranker: Whether to apply the Nemotron reranker.
            fusion_top_k: Number of candidates passed from fusion to reranking.
            custom_docs: Optional documents merged by ``pubkey`` before indexing.

        Returns:
            A pair ``(document_count, preview_rows)`` for the Streamlit UI.
        """
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
            reranker=RerankerConfig(
                name="nemotron" if enable_reranker else None, enabled=enable_reranker
            ),
            use_fusion=enable_fusion,
            fusion_top_k=fusion_top_k,
            sparse_cache_top_k=2000,
            final_top_k=5,
        )

        base_docs = load_dataset(
            CHECKTHAT_DATASET, "collection", split="collection"
        ).to_list()
        self.pipeline = build_pipeline_from_config(config)
        self.pipeline.index_collection(base_docs, custom_documents=custom_docs)

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
    def search(self, query_text: str):
        """Search a tweet against the already-indexed demo pipeline.

        Raises:
            RuntimeError: If the Modal container restarted and lost its
                in-memory pipeline state.
        """
        if not hasattr(self, "pipeline") or self.pipeline is None:
            raise RuntimeError(
                "Pipeline not initialized. The container may have restarted. Please index first."
            )

        result = self.pipeline.search_text(query_text, lang="auto")
        preds = result["preds"]
        docs_by_pubkey = {
            doc["pubkey"]: doc for doc in self.pipeline.collection_documents
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
