"""Dense retrieval helpers."""

from __future__ import annotations

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def topk_indices(similarities: np.ndarray, k: int) -> list[int]:
    scores = np.asarray(similarities)
    if scores.size == 0 or k <= 0:
        return []

    k = min(k, scores.shape[0])
    if k == scores.shape[0]:
        return np.argsort(-scores).tolist()

    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist()


def retrieve_top_pubkeys(
    query_embedding: np.ndarray,
    paper_embeddings: np.ndarray,
    pubkeys: list[str],
    k: int,
) -> list[str]:
    if not pubkeys or k <= 0:
        return []

    query_norm = l2_normalize(np.asarray(query_embedding, dtype=float)).reshape(-1)
    paper_norm = l2_normalize(np.asarray(paper_embeddings, dtype=float))
    if paper_norm.shape[0] != len(pubkeys):
        raise ValueError(
            "Length mismatch between paper embeddings and pubkeys. "
            "Rebuild index cache to restore consistency."
        )
    if paper_norm.shape[1] != query_norm.shape[0]:
        raise ValueError("Embedding dimension mismatch between query and paper embeddings.")
    similarities = paper_norm @ query_norm

    ranked_indices = topk_indices(similarities, k=k)
    return [pubkeys[index] for index in ranked_indices]
