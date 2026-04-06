from .interfaces import BaseRetriever


class BGEM3Retriever(BaseRetriever):
    def __init__(self, model_name: str = "BAAI/bge-m3", lora_id: str = None):
        # Heavy imports happen safely inside the cloud!
        import os

        import torch
        from peft import PeftModel
        from sentence_transformers import SentenceTransformer, util

        self.util = util
        self.torch = torch

        print(f"Loading Dense Retriever ({model_name})...")
        self.model = SentenceTransformer(model_name, device="cuda")

        if lora_id:
            print(f"Injecting LoRA adapters from {lora_id}...")
            hf_token = os.environ.get("HF_TOKEN")
            self.model[0].auto_model = PeftModel.from_pretrained(
                self.model[0].auto_model, lora_id, token=hf_token
            )
            self.model = self.model.to("cuda")

    def index(self, corpus: list[str]):
        print("Generating dense embeddings for the collection...")
        self.embeddings = self.model.encode_document(
            corpus, convert_to_tensor=True, show_progress_bar=True, device="cuda"
        )

    def search(self, query: str) -> list[int]:
        query_embedding = self.model.encode_query(query, convert_to_tensor=True)
        scores = self.util.cos_sim(query_embedding, self.embeddings)[0]
        return self.torch.argsort(scores, descending=True).tolist()


class BM25Retriever(BaseRetriever):
    def __init__(self):
        print("Initializing BM25 Sparse Retriever...")
        self.bm25_model = None

    def index(self, corpus: list[str]):
        from rank_bm25 import BM25Okapi

        print("Generating BM25 sparse index...")
        tokenized_corpus = [text.lower().split() for text in corpus]
        self.bm25_model = BM25Okapi(tokenized_corpus)

    def search(self, query: str) -> list[int]:
        import numpy as np

        tokenized_query = query.lower().split()
        scores = self.bm25_model.get_scores(tokenized_query)
        return np.argsort(scores)[::-1].tolist()
