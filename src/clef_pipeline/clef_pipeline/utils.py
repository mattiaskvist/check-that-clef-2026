"""Shared constants and utility functions for retrieval and scoring."""

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

CHECKTHAT_DATASET = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"

STOPWORDS = frozenset(
    stopwords.words("english") + stopwords.words("german") + stopwords.words("french")
)


def article_to_text(doc: dict) -> str:
    """Convert a document dict into minimal title/abstract text."""
    return f"{doc['title']}\n{doc['abstract']}"


def MRR_at_5(preds: list[str], label: str) -> float:
    """Compute reciprocal rank at cutoff 5 for one query.

    Args:
        preds: Ranked predicted publication keys.
        label: Ground-truth publication key.

    Returns:
        Reciprocal rank at 5, or ``0.0`` if not found.
    """
    preds = preds[:5]
    if label in preds:
        return 1 / (preds.index(label) + 1)
    else:
        return 0.0


def recall_at_K(predictions: list[str], target: str, k: int) -> float:
    """Calculate hit-rate style recall at ``k`` for a single query.

    Args:
        predictions: Ranked predicted publication keys.
        target: Ground-truth publication key.
        k: Cutoff rank.

    Returns:
        ``1.0`` if target appears within top-``k``, else ``0.0``.
    """
    return 1.0 if target in predictions[:k] else 0.0


class FusionProcessor:
    """Fuse multiple ranked lists into one final candidate ordering."""

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: list[list[int]], k: int = 60, top_k: int = 10
    ) -> list[int]:
        """Apply Reciprocal Rank Fusion (RRF) over ranked result lists.

        Args:
            ranked_lists: Ranked document-id lists from multiple retrievers.
            k: RRF rank dampening constant.
            top_k: Number of fused document ids to return.

        Returns:
            Fused document ids sorted by descending RRF score.
        """
        rrf_scores = {}
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in sorted_docs[:top_k]]
