import json

import streamlit as st
from datasets import load_dataset

from .pipeline_config import PipelineConfig, RerankerConfig, RetrieverConfig
from .registry import build_pipeline_from_config
from .utils import CHECKTHAT_DATASET


def _build_runtime_config(
    selected_retrievers: list[str],
    enable_fusion: bool,
    enable_reranker: bool,
    fusion_top_k: int,
):
    return PipelineConfig(
        retrievers=[RetrieverConfig(name=name) for name in selected_retrievers],
        reranker=RerankerConfig(
            name="nemotron" if enable_reranker else None,
            enabled=enable_reranker,
        ),
        use_fusion=enable_fusion,
        fusion_top_k=fusion_top_k,
        sparse_cache_top_k=2000,
        final_top_k=5,
    )


@st.cache_data(show_spinner=False)
def _load_base_documents() -> list[dict]:
    return load_dataset(CHECKTHAT_DATASET, "collection", split="collection").to_list()


def _parse_custom_docs(raw_json: str) -> list[dict]:
    if not raw_json.strip():
        return []
    parsed = json.loads(raw_json)
    if not isinstance(parsed, list):
        raise ValueError("Custom documents JSON must be a list.")
    required = {"pubkey", "title", "abstract"}
    for idx, doc in enumerate(parsed):
        if not isinstance(doc, dict):
            raise ValueError(f"Custom document at index {idx} must be an object.")
        missing = required - set(doc)
        if missing:
            raise ValueError(f"Custom document at index {idx} is missing fields: {sorted(missing)}")
    return parsed


def _index_pipeline(config: PipelineConfig, custom_docs: list[dict]):
    base_docs = _load_base_documents()
    pipeline = build_pipeline_from_config(config)
    pipeline.index_collection(base_docs, custom_documents=custom_docs)
    return pipeline


def main():
    st.set_page_config(page_title="CLEF Source Retrieval Demo", layout="wide")
    st.title("CLEF 2026 Source Retrieval Demo")
    st.write(
        "Index the collection, optionally merge custom documents, and retrieve top-5 article matches for your tweet."
    )

    with st.sidebar:
        st.header("Pipeline Settings")
        retrievers = st.multiselect(
            "Retrievers",
            options=["harrier-270m", "sparse", "harrier-27b", "bge-m3"],
            default=["harrier-270m", "sparse"],
        )
        enable_fusion = st.checkbox("Use RRF fusion", value=True)
        enable_reranker = st.checkbox("Use Nemotron reranker", value=True)
        fusion_top_k = st.slider("Fusion candidate top-k", min_value=5, max_value=100, value=30, step=5)
        st.markdown("### Custom documents (JSON list)")
        custom_docs_json = st.text_area(
            "Optional documents to add/override by pubkey",
            placeholder='[{"pubkey":"custom-1","title":"...","abstract":"...","authors":"...","venue":"..."}]',
            height=180,
        )

    if not retrievers:
        st.warning("Select at least one retriever.")
        return

    config = _build_runtime_config(
        selected_retrievers=retrievers,
        enable_fusion=enable_fusion,
        enable_reranker=enable_reranker,
        fusion_top_k=fusion_top_k,
    )

    if st.button("Index documents", type="primary"):
        custom_docs = _parse_custom_docs(custom_docs_json)
        with st.spinner("Indexing corpus..."):
            pipeline = _index_pipeline(config, custom_docs)
            st.session_state["pipeline"] = pipeline
            st.session_state["indexed_docs"] = pipeline.collection_documents
        st.success(f"Indexed {len(st.session_state['indexed_docs'])} documents.")

    indexed_docs = st.session_state.get("indexed_docs")
    if indexed_docs:
        st.subheader("Indexed documents")
        st.dataframe(
            [
                {
                    "pubkey": doc.get("pubkey"),
                    "title": doc.get("title"),
                    "venue": doc.get("venue"),
                }
                for doc in indexed_docs
            ],
            use_container_width=True,
            height=320,
        )

    st.subheader("Query")
    query_text = st.text_area("Write your tweet", height=120)
    query_lang = st.selectbox("Tweet language", options=["en", "de", "fr"], index=0)

    if st.button("Find top-5 matches"):
        pipeline = st.session_state.get("pipeline")
        if pipeline is None:
            st.error("Index documents first.")
            return
        if not query_text.strip():
            st.error("Enter a tweet first.")
            return

        with st.spinner("Searching..."):
            result = pipeline.search_text(query_text, lang=query_lang)
            preds = result["preds"]
            docs_by_pubkey = {doc["pubkey"]: doc for doc in pipeline.collection_documents}
            rows = []
            for rank, pubkey in enumerate(preds, start=1):
                doc = docs_by_pubkey.get(pubkey, {})
                rows.append(
                    {
                        "rank": rank,
                        "pubkey": pubkey,
                        "title": doc.get("title", ""),
                        "abstract": doc.get("abstract", ""),
                    }
                )
            st.write("### Top-5 matches")
            st.dataframe(rows, use_container_width=True)
            st.write("### Stage outputs")
            st.json(result["stages"])


if __name__ == "__main__":
    main()
