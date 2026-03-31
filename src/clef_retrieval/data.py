"""Dataset loading helpers."""

from collections.abc import Sequence

from datasets import load_dataset


DATASET_ID = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"


def normalize_authors(authors: object) -> str:
    if isinstance(authors, str):
        return authors.strip()
    if isinstance(authors, Sequence):
        values = [str(author).strip() for author in authors]
        return "; ".join(value for value in values if value)
    return ""


def load_collection():
    return load_dataset(DATASET_ID, "collection")["collection"]


def load_language_split(lang: str, split: str):
    return load_dataset(DATASET_ID, lang)[split]
