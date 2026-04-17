import json
import time

import modal
import pandas as pd
import streamlit as st


@st.cache_resource
def get_backend():
    return modal.Cls.from_name("clef-backend", "PipelineBackend")


def _parse_custom_docs(raw_json: str) -> list[dict]:
    if not raw_json.strip():
        return
    parsed = json.loads(raw_json)
    if not isinstance(parsed, list):
        raise ValueError("Custom documents JSON must be a list.")
    required = {"pubkey", "title", "abstract"}
    for idx, doc in enumerate(parsed):
        if not isinstance(doc, dict):
            raise ValueError(f"Custom document at index {idx} must be an object.")
        missing = required - set(doc)
        if missing:
            raise ValueError(
                f"Custom document at index {idx} is missing fields: {sorted(missing)}"
            )
    return parsed


# Refactored Polling Architecture:
# We use st.fragment to run the check every 2 seconds without blocking Streamlit's main thread.
@st.fragment(run_every=2)
def asynchronous_task_monitor():
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
        st.info(f"⏳ Indexing on GPU cluster... Please wait ({elapsed_time}s elapsed)")


def main():
    st.set_page_config(page_title="CLEF Source Retrieval Demo", layout="wide")
    st.title("CLEF 2026 Source Retrieval Demo")
    st.write(
        "Index the collection, optionally merge custom documents, and retrieve top-5 article matches for your tweet."
    )

    PipelineBackend = get_backend()

    with st.sidebar:
        st.header("Pipeline Settings")
        retrievers = st.multiselect(
            "Retrievers",
            options=["harrier-270m", "sparse", "harrier-27b", "bge-m3"],
            default=["harrier-270m", "sparse"],
        )
        enable_fusion = st.checkbox("Use RRF fusion", value=True)
        enable_reranker = st.checkbox("Use Nemotron reranker", value=True)
        fusion_top_k = st.slider(
            "Fusion candidate top-k", min_value=5, max_value=100, value=30, step=5
        )
        st.markdown("### Custom documents (JSON list)")
        custom_docs_json = st.text_area(
            "Optional documents to add/override by pubkey",
            placeholder='[{"pubkey":"custom-1","title":"...","abstract":"...","authors":"...","venue":"..."}]',
            height=180,
        )

    if not retrievers:
        st.warning("Select at least one retriever.")
        return

    if st.button("Index documents", type="primary"):
        custom_docs = _parse_custom_docs(custom_docs_json)

        try:
            # Non-blocking remote procedure call
            call = PipelineBackend().index_documents.spawn(
                selected_retrievers=retrievers,
                enable_fusion=enable_fusion,
                enable_reranker=enable_reranker,
                fusion_top_k=fusion_top_k,
                custom_docs=custom_docs,
            )

            # Preserve state and initiate the polling fragment
            st.session_state["active_gpu_call"] = call
            st.session_state["start_time"] = time.time()
            st.session_state["is_polling"] = True

        except Exception as e:
            st.error(f"Failed to initiate remote GPU indexer: {e}")

    # Render the polling fragment if an indexing job is active
    if st.session_state.get("is_polling", False):
        asynchronous_task_monitor()

    # --- UI RENDERED AFTER INDEXING ---
    # We ensure this doesn't accidentally trigger while polling is still active
    if st.session_state.get("is_indexed", False) and not st.session_state.get(
        "is_polling", False
    ):
        actual_count = st.session_state.get("doc_count", 0)

        st.success(f"Successfully indexed {actual_count} documents on GPU!")

        if actual_count > 50:
            st.warning(
                f"Data payload truncated from {actual_count} to 50 rows to prevent WebSocket saturation."
            )

        st.subheader("Indexed documents preview (Bounded)")

        if not st.session_state.get("docs_summary"):
            st.warning("The preview list is empty. Check backend data.")
        else:
            df = pd.DataFrame(st.session_state["docs_summary"])
            st.dataframe(df, width="stretch", height=320)

        st.subheader("Query")
        query_text = st.text_area("Write your tweet", height=120)

        if st.button("Find top-5 matches"):
            if not query_text.strip():
                st.error("Enter a tweet first.")
                return

            with st.spinner("Routing search to GPU..."):
                try:
                    # Depending on how long search takes, you could apply the fragment architecture
                    # here too, but normally search returns within seconds rather than minutes.
                    results = PipelineBackend().search.remote(query_text)

                    st.write("### Top-5 matches")
                    st.dataframe(pd.DataFrame(results["rows"]), width="stretch")
                    st.write("### Stage outputs")
                    st.json(results["stages"])
                except Exception as e:
                    st.error(f"Search failed: {e}")


if __name__ == "__main__":
    main()
