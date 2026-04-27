"""Streamlit frontend for interactive CLEF source retrieval."""

import time

import modal
import streamlit as st


@st.cache_resource
def get_backend():
    """Return the deployed Modal backend class used by the demo UI."""
    return modal.Cls.from_name("clef-backend", "PipelineBackend")


# Refactored Polling Architecture:
# We use st.fragment to run the check every 2 seconds without blocking Streamlit's main thread.
@st.fragment(run_every=2)
def asynchronous_task_monitor():
    """Poll the asynchronous Modal indexing call without blocking Streamlit."""
    if not st.session_state.get("is_polling", False):
        return

    start_time = st.session_state.get("start_time", time.time())
    active_call = st.session_state.get("active_gpu_call")

    if active_call is None:
        st.error("Lost reference to the remote GPU task.")
        st.session_state["is_polling"] = False
        return

    try:
        # Rapid, non-blocking check against the task future
        result = active_call.get(timeout=0.1)

        doc_count, docs_summary = result

        # Defensive Payload Bounding:
        # Cap the rows so the Arrow serialization never exceeds 2 MiB
        MAX_SAFE_RENDER_ROWS = 50
        if docs_summary and len(docs_summary) > MAX_SAFE_RENDER_ROWS:
            safe_summary = docs_summary[:MAX_SAFE_RENDER_ROWS]
        else:
            safe_summary = docs_summary if docs_summary else []

        st.session_state["docs_summary"] = safe_summary
        st.session_state["doc_count"] = doc_count
        st.session_state["is_indexed"] = True

        # Terminate polling mode
        st.session_state["is_polling"] = False

        # Trigger a full UI rerun to render the table and query widgets
        st.rerun()

    except TimeoutError:
        # Task is still running on the GPU. Yield the thread.
        elapsed_time = int(time.time() - start_time)
        st.info(
            f"""
            **Preparing retrieval system**

            Loading cached embeddings onto the GPU for semantic search.

            ⏱ Elapsed time: **{elapsed_time}s**
            """
        )


def truncate(text, limit=100):
    """Shorten long document metadata for compact card rendering."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def render_result_card(row, rank):
    """Render one ranked retrieval result as a styled Streamlit card."""
    title = row.get("title", "Untitled")
    authors = row.get("authors", "")
    abstract = row.get("abstract", "")
    score = row.get("score", None)
    pubkey = row.get("pubkey", row.get("doc_id", "unknown"))

    score_html = f'<span class="result-score">score {score:.3f}</span>' if score else ""
    pubkey_html = f'<span class="pubkey-badge">id {pubkey}</span>'

    st.markdown(
        f"""
        <div class="result-card">

        <div class="result-title">
        #{rank} {title}
        {score_html}
        {pubkey_html}
        </div>

        <div class="result-meta">
        <strong>Authors:</strong> {authors}
        </div>

        <div class="result-abstract">
        {abstract}
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stage_outputs(stages, top_ids, enable_fusion=True):
    """Render dense, sparse, fusion, and reranker stage outputs side by side."""
    dense = stages.get("dense", [])[:100]
    sparse = stages.get("sparse", [])[:100]
    rrf = stages.get("rrf", []) if enable_fusion else []
    final = stages.get("final", [])

    st.write("### Retrieval stages")

    c1, c2, c3, c4 = st.columns(4)

    def render_stage(col, title, data):
        with col:
            st.markdown(
                f'<div class="stage-title">{title}</div>', unsafe_allow_html=True
            )

            lines = []

            for item in data:
                # Handle both formats:
                # dense/sparse: "0 - 99"
                # rrf/final: "0:4553"
                doc_id = None

                if isinstance(item, str) and ":" in item:
                    doc_id = item.split(":")[1]
                elif isinstance(item, int):
                    doc_id = item
                else:
                    doc_id = item

                if str(doc_id) in {str(x) for x in top_ids}:
                    lines.append(f'<span class="stage-hit">{item}</span>')
                else:
                    lines.append(f'<span class="stage-miss">{item}</span>')

            text = "<br>".join(lines)

            st.markdown(
                f'<div class="stage-box">{text}</div>',
                unsafe_allow_html=True,
            )

    render_stage(c1, "Dense", dense)
    render_stage(c2, "Sparse", sparse)
    render_stage(c3, "Fusion", rrf)
    render_stage(c4, "Reranker", final)


def render_doc_preview(doc, idx):
    """Render one collection document preview card."""
    import html

    title = html.escape(doc.get("title", "Untitled"))
    authors = html.escape(truncate(doc.get("authors", ""), 100))
    abstract = html.escape(truncate(doc.get("abstract", ""), 180))
    pubkey = doc.get("pubkey", doc.get("doc_id", "unknown"))

    st.markdown(
        f"""
        <div class="doc-preview-card">

        <div class="doc-preview-title">
        {idx}. {title}
        <span class="doc-id">id {pubkey}</span>
        </div>

        <div class="doc-preview-authors">
        {authors}
        </div>

        <div class="doc-preview-abstract">
        {abstract}
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Render the Streamlit app and route user actions to the Modal backend."""
    st.set_page_config(page_title="CLEF Source Retrieval Demo", layout="wide")
    st.markdown(
        """
        <style>

        .result-card {
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 18px;
            background-color: #0f172a;
            border: 1px solid #1e293b;
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        }

        .result-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .result-meta {
            font-size: 13px;
            color: #94a3b8;
            margin-bottom: 10px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .result-abstract {
            font-size: 14px;
            line-height: 1.5;
        }

        .result-score {
            float: right;
            font-size: 13px;
            background: #1e293b;
            padding: 4px 10px;
            border-radius: 6px;
        }

        .stage-box {
            border-radius: 10px;
            border: 1px solid #1e293b;
            padding: 12px;
            background: #020617;
            height: 320px;
            overflow-y: auto;
            font-size: 13px;
        }

        .stage-title {
            font-weight: 600;
            margin-bottom: 8px;
        }

        .pubkey-badge {
            float: right;
            font-size: 12px;
            background: #020617;
            padding: 4px 8px;
            border-radius: 6px;
            margin-right: 8px;
            border: 1px solid #1e293b;
            color: #94a3b8;
            font-family: monospace;
        }

        .stage-hit {
            color: #22c55e;
            font-weight: 600;
        }

        .stage-miss {
            color: #94a3b8;
        }

        .doc-preview-card {
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 6px;
            background: #020617;
            border: 1px solid #1e293b;
        }

        .doc-preview-title {
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 2px;
        }

        .doc-preview-authors {
            font-size: 11px;
            color: #94a3b8;
            margin-bottom: 2px;
        }

        .doc-preview-abstract {
            font-size: 12px;
            line-height: 1.25;
            color: #cbd5f5;
        }

        .doc-id {
            float: right;
            font-family: monospace;
            font-size: 10px;
            background: #0f172a;
            padding: 1px 5px;
            border-radius: 4px;
            color: #94a3b8;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("CLEF 2026 Source Retrieval Demo")
    st.write(
        "Load precomputed embeddings from the cache and retrieve top-5 article matches for your tweet."
    )

    PipelineBackend = get_backend()

    with st.sidebar:
        st.header("Pipeline Settings")

        dense_model = st.selectbox(
            "Dense model",
            options=["none", "harrier-270m", "bge-m3"],
            index=1,
        )

        use_sparse = st.checkbox("Enable sparse retriever", value=True)

        retrievers = []

        if dense_model != "none":
            retrievers.append(dense_model)
        if use_sparse:
            retrievers.append("sparse")

        fusion_method = st.selectbox(
            "Fusion method",
            options=["none", "rrf", "random_forest"],
            index=1,
        )

        enable_fusion = fusion_method != "none"

        fusion_top_k = st.slider(
            "Fusion candidate top-k",
            min_value=5,
            max_value=100,
            value=30,
            step=5,
            disabled=not enable_fusion,
        )

        reranker_name = st.selectbox(
            "Reranker",
            options=["none", "nemotron", "qwen3-reranker-8b"],
            index=1,
        )

    if not retrievers:
        st.warning("Select at least one retriever.")
        return

    prev_retrievers = st.session_state.get("selected_retrievers")
    if prev_retrievers is not None and set(prev_retrievers) != set(retrievers):
        st.session_state["is_indexed"] = False
        st.session_state["docs_summary"] = []
        st.session_state["doc_count"] = 0
    st.session_state["selected_retrievers"] = list(retrievers)

    if st.button("Load cached embeddings", type="primary"):
        try:
            # Non-blocking remote procedure call
            call = PipelineBackend().load_cached_collection.spawn(
                selected_retrievers=retrievers,
            )

            # Preserve state and initiate the polling fragment
            st.session_state["active_gpu_call"] = call
            st.session_state["start_time"] = time.time()
            st.session_state["is_polling"] = True

        except Exception as e:
            st.error(f"Failed to initiate remote cache loader: {e}")

    # Render the polling fragment if an indexing job is active
    if st.session_state.get("is_polling", False):
        asynchronous_task_monitor()

    # We ensure this doesn't accidentally trigger while polling is still active
    if st.session_state.get("is_indexed", False) and not st.session_state.get(
        "is_polling", False
    ):
        actual_count = st.session_state.get("doc_count", 0)

        st.success(
            f"""
            **Retrieval system ready**

            Indexed {actual_count:,} documents with precomputed embeddings.

            You can now search for matching articles.
            """
        )

        st.subheader("Documents preview")

        if not st.session_state.get("docs_summary"):
            st.warning("The preview list is empty. Check backend data.")
        else:
            with st.container(height=300):
                docs = st.session_state["docs_summary"]

                for i, doc in enumerate(docs):
                    render_doc_preview(doc, i + 1)

        col1, col2 = st.columns([1, 2])
        results = st.session_state.get("results")
        top_ids = st.session_state.get("top_ids")

        with col1:
            st.subheader("Query")

            query_text = st.text_area(
                "Enter a tweet or claim to search for relevant articles.",
                height=160,
                placeholder="Example: 'RNNs are great for volatility forecasting!'",
            )

            search_clicked = st.button("Find top-5 matches", use_container_width=True)

        with col2:
            st.subheader("Results")

            if search_clicked:
                if not query_text.strip():
                    st.error("Enter a tweet first.")
                    st.stop()

                with st.spinner("Searching articles and ranking results..."):
                    try:
                        results = PipelineBackend().search.remote(
                            query_text,
                            enable_fusion=enable_fusion,
                            fusion_method=fusion_method,
                            fusion_top_k=fusion_top_k,
                            reranker_name=reranker_name,
                        )

                        rows = results["rows"]

                        if not rows:
                            st.warning(
                                "No relevant articles were found. Try rephrasing the tweet or claim."
                            )
                        else:
                            for i, row in enumerate(rows, start=1):
                                render_result_card(row, i)

                        top_ids = {row.get("pubkey", row.get("doc_id")) for row in rows}
                        st.session_state["results"] = results
                        st.session_state["top_ids"] = top_ids

                    except Exception as e:
                        st.error(f"Search failed: {e}")

        if st.session_state.get("results"):
            results = st.session_state["results"]
            top_ids = st.session_state["top_ids"]

            st.markdown("---")
            render_stage_outputs(
                results["stages"], top_ids, enable_fusion=enable_fusion
            )


if __name__ == "__main__":
    main()
