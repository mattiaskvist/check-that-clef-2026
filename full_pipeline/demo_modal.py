import subprocess

import modal

app = modal.App("checkthat-streamlit-demo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "streamlit",
        "sentence-transformers",
        "peft",
        "rank_bm25",
        "datasets",
        "tqdm",
        "transformers",
        "accelerate",
        "nltk",
        "Pillow",
        "torchvision",
        "deep-translator",
    )
    .add_local_python_source("full_pipeline")
)

embedding_cache = modal.Volume.from_name("checkthat-embedding-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 12,
    volumes={"/cache/embeddings": embedding_cache},
    secrets=[modal.Secret.from_name("hf-token")],
)
@modal.web_server(8501)
def serve_streamlit():
    subprocess.Popen(
        [
            "streamlit",
            "run",
            "/root/full_pipeline/demo_app.py",
            "--server.port",
            "8501",
            "--server.address",
            "0.0.0.0",
            "--server.headless",
            "true",
        ]
    )
