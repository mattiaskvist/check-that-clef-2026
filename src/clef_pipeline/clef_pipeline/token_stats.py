"""CPU-only token length statistics for CheckThat documents and queries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from .utils import (
    CHECKTHAT_DATASET,
    article_to_text,
    load_query_split,
    read_custom_papers,
)

# Matches the default Qwen3 reranker in ``clef_pipeline.rerankers.Qwen3Reranker``.
DEFAULT_TOKENIZER_MODEL_ID = "Qwen/Qwen3-Reranker-8B"

PERCENTILES = (50.0, 90.0, 95.0, 99.0, 99.5, 99.9)


def load_docs_from_jsonl(path: Path) -> list[dict]:
    """Load one JSON object per line; each row must include title and abstract."""
    docs: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "title" not in row or "abstract" not in row:
                raise ValueError(
                    f"{path}:{line_no}: expected keys 'title' and 'abstract'"
                )
            docs.append(row)
    return docs


def load_collection_documents(
    *,
    jsonl_path: Path | str | None = None,
    custom_papers_path: Path | str | None = None,
) -> list[dict]:
    """Load base collection from JSONL or Hugging Face, then optional CSV append."""
    if jsonl_path is not None:
        docs = load_docs_from_jsonl(Path(jsonl_path))
    else:
        from datasets import load_dataset

        collection = load_dataset(
            CHECKTHAT_DATASET, "collection", split="collection"
        )
        docs = collection.to_list()
    if custom_papers_path is not None:
        docs.extend(read_custom_papers(str(custom_papers_path)))
    return docs


def load_collection_texts(
    *,
    jsonl_path: Path | str | None = None,
    custom_papers_path: Path | str | None = None,
) -> list[str]:
    """Load collection documents and return ``title\\nabstract`` per doc."""
    docs = load_collection_documents(
        jsonl_path=jsonl_path, custom_papers_path=custom_papers_path
    )
    return [article_to_text(doc) for doc in docs]


def load_query_texts(languages: list[str], split: str) -> list[str]:
    """Load raw query texts for one or more languages from a given split."""
    texts: list[str] = []
    for lang in languages:
        rows = load_query_split(lang, split)
        texts.extend(str(row["text"]) for row in rows)
    return texts


def compute_token_length_summary(
    texts: list[str],
    tokenizer_id: str,
    batch_size: int = 64,
    label: str = "documents",
) -> dict[str, Any]:
    """Tokenize texts and return min, percentile, and max token counts."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    lengths: list[int] = []
    bs = max(1, batch_size)
    for start in tqdm(range(0, len(texts), bs), desc="Tokenizing", unit="batch"):
        batch = texts[start : start + bs]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])

    arr = np.asarray(lengths, dtype=np.int64)
    if arr.size == 0:
        raise ValueError("No texts to tokenize.")
    percentile_values = np.percentile(arr, list(PERCENTILES))
    summary: dict[str, Any] = {
        "label": label,
        "count": int(arr.size),
        "tokenizer": tokenizer_id,
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }
    for percentile, value in zip(PERCENTILES, percentile_values):
        summary[_percentile_key(percentile)] = float(value)
    return summary


def _percentile_key(percentile: float) -> str:
    """Format ``99.0`` as ``p99`` and ``99.5`` as ``p99.5``."""
    if float(percentile).is_integer():
        return f"p{int(percentile)}"
    return f"p{percentile:g}"


def print_token_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{summary['label']}: {summary['count']}")
    print(f"tokenizer: {summary['tokenizer']}")
    print(f"min:    {summary['min']}")
    print(f"mean:   {summary['mean']:.1f}")
    for percentile in PERCENTILES:
        key = _percentile_key(percentile)
        print(f"{key + ':':<7} {summary[key]:.1f}")
    print(f"max:    {summary['max']}")
