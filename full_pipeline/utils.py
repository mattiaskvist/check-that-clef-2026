import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

CHECKTHAT_DATASET = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"

STOPWORDS = frozenset(
    stopwords.words("english") + stopwords.words("german") + stopwords.words("french")
)


def article_to_text(doc: dict) -> str:
    return f"{doc['title']}\n{doc['abstract']}"


def MRR_at_5(preds: list[str], label: str) -> float:
    preds = preds[:5]
    if label in preds:
        return 1 / (preds.index(label) + 1)
    else:
        return 0.0


class FusionProcessor:
    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: list[list[int]], k: int = 60, top_k: int = 10
    ) -> tuple[list[int], dict[int, float]]:
        """Return (ranked_doc_ids, {doc_id: rrf_score}) for the top-k candidates."""
        rrf_scores = {}
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:top_k]
        top_ids = [doc_id for doc_id, score in top_docs]
        top_scores = {doc_id: score for doc_id, score in top_docs}
        return top_ids, top_scores
