"""Shared constants and utility functions for retrieval and scoring."""

import nltk
from nltk.corpus import stopwords
import csv
from pathlib import Path

nltk.download("stopwords", quiet=True)

CHECKTHAT_DATASET = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"

STOPWORDS = frozenset(
    stopwords.words("english") + stopwords.words("german") + stopwords.words("french")
)


def load_query_split(lang: str, split: str) -> list[dict]:
    """Load one language split from the official CheckThat dataset.

    The competition dataset exposes ``train`` and ``dev`` through the language
    configuration, while ``test`` is exposed via a top-level ``test`` config
    containing one split per language.

    Args:
        lang: Language code such as ``en``, ``de``, or ``fr``.
        split: Requested split name.

    Returns:
        Query rows as plain dictionaries.
    """
    from datasets import load_dataset

    if split == "test":
        return list(load_dataset(CHECKTHAT_DATASET, name="test")[lang])
    return list(load_dataset(CHECKTHAT_DATASET, lang)[split])


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


def read_custom_papers(file_path: str, start_id: int = 11000) -> list[dict]:
    """Read custom papers from a CSV file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"custom_papers file not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("custom_papers must be a .csv file for now")

    documents = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            pubkey = start_id + i
            documents.append(
                {
                    "pubkey": pubkey,
                    "title": row.get("Title", "").strip(),
                    "authors": row.get("Authors", "").strip(),
                    "venue": row.get("Venue", "").strip(),
                    "abstract": row.get("Abstract", "").strip(),
                }
            )

    return documents


def translate_de_tweets_to_fr(texts: list[str]) -> list[str]:
    """Translate German tweet/query texts into French.

    Notes:
        - Intended for query preprocessing only (tweets), not for collection articles.
        - Uses ``deep_translator`` backends which may require network access.
        - On any translation failure, returns the original texts unchanged.
    """
    if not texts:
        return []
    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="de", target="fr")
        translate_batch = getattr(translator, "translate_batch", None)
        if callable(translate_batch):
            translated = translate_batch(texts)
        else:
            translated = [translator.translate(text) for text in texts]
        return [t if isinstance(t, str) and t else s for s, t in zip(texts, translated)]
    except Exception:
        return list(texts)


def translate_de_articles_to_fr(docs: list[dict]) -> list[dict]:
    """Translate German collection documents (title/abstract) into French.

    Only documents explicitly tagged as German are translated. This function
    expects language metadata to be present (e.g. ``lang`` or ``language``).

    Translation is best-effort: on any failure, the original documents are
    returned unchanged.
    """
    if not docs:
        return []

    indices: list[int] = []
    titles: list[str] = []
    abstracts: list[str] = []
    for i, doc in enumerate(docs):
        lang = str(doc.get("lang") or doc.get("language") or "").lower().strip()
        if not (lang == "de" or lang.startswith("de-")):
            continue
        indices.append(i)
        titles.append(str(doc.get("title") or ""))
        abstracts.append(str(doc.get("abstract") or ""))

    if not indices:
        return list(docs)

    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="de", target="fr")
        translate_batch = getattr(translator, "translate_batch", None)
        if callable(translate_batch):
            titles_fr = translate_batch(titles)
            abstracts_fr = translate_batch(abstracts)
        else:
            titles_fr = [translator.translate(text) for text in titles]
            abstracts_fr = [translator.translate(text) for text in abstracts]

        updated = [dict(doc) for doc in docs]
        for pos, idx in enumerate(indices):
            title_fr = titles_fr[pos] if isinstance(titles_fr[pos], str) else titles[pos]
            abstract_fr = (
                abstracts_fr[pos]
                if isinstance(abstracts_fr[pos], str)
                else abstracts[pos]
            )
            updated[idx]["title"] = title_fr or titles[pos]
            updated[idx]["abstract"] = abstract_fr or abstracts[pos]
        return updated
    except Exception:
        return list(docs)


def _write_recall_log(
    output_file: str,
    *,
    summary: dict[str, object],
    fusion_label: str,
) -> None:
    import json

    payload = {
        "fusion_label": fusion_label,
        "recall_cutoffs": summary.get("recall_cutoffs", []),
        "languages": summary.get("languages", {}),
        "global": summary.get("global", None),
        "total_queries": summary.get("total_queries", 0),
        "total_labeled_queries": summary.get("total_labeled_queries", 0),
        "total_unlabeled_queries": summary.get("total_unlabeled_queries", 0),
    }
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)


def _print_all_recalls(summary: dict[str, object], *, fusion_label: str) -> None:
    cutoffs = summary.get("recall_cutoffs") or []
    global_metrics = summary.get("global")
    if not cutoffs or global_metrics is None:
        return

    print("\n" + "=" * 80)
    print("ALL RECALLS (GLOBAL AVERAGE)")
    print("=" * 80)
    for k in cutoffs:
        key = f"r{k}"
        dense = global_metrics["dense"].get(key, 0.0)
        sparse = global_metrics["sparse"].get(key, 0.0)
        fused = global_metrics["rrf"].get(key, 0.0)
        final = global_metrics["final"].get(key, 0.0)
        print(
            f"R@{k:<3} | Dense: {dense:.4f} | Sparse: {sparse:.4f} | {fusion_label}: {fused:.4f} | Final: {final:.4f}"
        )
