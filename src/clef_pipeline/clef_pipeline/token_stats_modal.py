"""Modal wrapper for token length statistics."""

from __future__ import annotations

import modal

from .token_stats import (
    DEFAULT_TOKENIZER_MODEL_ID,
    compute_token_length_summary,
    load_collection_texts,
    load_query_texts,
    print_token_summary,
)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets",
    "transformers",
    "numpy",
    "tqdm",
    "nltk",
)

app = modal.App("checkthat-token-stats")


@app.function(
    image=image,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("hf-token")],
)
def compute_stats_remote(
    mode: str = "collection",
    split: str = "test",
    languages: list[str] | None = None,
    tokenizer: str = DEFAULT_TOKENIZER_MODEL_ID,
    batch_size: int = 64,
) -> dict:
    """Compute token stats on Modal and return summary dict."""
    langs = languages or ["de", "fr", "en"]
    if mode == "collection":
        texts = load_collection_texts()
        return compute_token_length_summary(
            texts,
            tokenizer,
            batch_size=batch_size,
            label="documents",
        )
    if mode == "queries":
        texts = load_query_texts(langs, split)
        return compute_token_length_summary(
            texts,
            tokenizer,
            batch_size=batch_size,
            label=f"queries[{split}/{'+'.join(langs)}]",
        )
    raise ValueError("mode must be one of: collection, queries")


@app.local_entrypoint()
def main(
    mode: str = "collection",
    split: str = "test",
    languages: str = "de,fr,en",
    tokenizer: str = DEFAULT_TOKENIZER_MODEL_ID,
    batch_size: int = 64,
):
    langs = [lang.strip() for lang in languages.split(",") if lang.strip()]
    summary = compute_stats_remote.remote(
        mode=mode,
        split=split,
        languages=langs,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    print_token_summary(summary)
