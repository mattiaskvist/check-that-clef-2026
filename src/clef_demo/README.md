# CLEF Demo

`src/clef_demo` contains the Streamlit demo for interactive source retrieval.
The UI runs as a Streamlit app, while indexing and search run in a Modal GPU
backend.

## Architecture

- `clef_demo.streamlit_app` renders the user interface.
- `clef_demo.backend` defines the Modal `PipelineBackend` class used for GPU
  indexing and search.
- `clef_demo.modal_app` hosts Streamlit itself as a Modal web server.

The Streamlit app calls:

```python
modal.Cls.from_name("clef-backend", "PipelineBackend")
```

That means the backend app must be deployed before the UI can perform indexing
or search.

## Prerequisites

From the repository root:

```bash
uv sync
uv run modal setup
uv run modal secret create hf-token HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

The backend reads the CheckThat dataset from Hugging Face and may load gated or
private models depending on selected retrievers and rerankers.

## Deploy The Backend

Deploy the GPU backend:

```bash
uv run modal deploy src/clef_demo/clef_demo/backend.py
```

This creates the Modal app `clef-backend` with class `PipelineBackend`.

## Run The UI Locally

After the backend is deployed, start Streamlit locally:

```bash
uv run streamlit run src/clef_demo/clef_demo/streamlit_app.py
```

Use the sidebar to select retrievers, fusion, and reranking. Click
`Index documents` before running queries.

## Serve The UI On Modal

For live development:

```bash
uv run modal serve -m clef_demo.modal_app
```

For a persistent deployment:

```bash
uv run modal deploy -m clef_demo.modal_app
```

Stop deployed demo apps when they are no longer needed:

```bash
uv run modal stop-app clef-backend
uv run modal stop-app checkthat-streamlit-demo
```

## Troubleshooting

- `Lost reference to the remote GPU task`: the Modal call reference was lost or
  the backend container restarted. Re-index documents.
- `Pipeline not initialized`: the backend scaled down or restarted after
  indexing. Click `Index documents` again.
- Search is unavailable until indexing finishes successfully.
- Large retrievers and rerankers can exceed the demo backend resource envelope.
  Prefer `harrier-270m` plus `sparse` for interactive use.
- The UI intentionally truncates the document preview to keep Streamlit payloads
  small. The full index remains in the backend.
