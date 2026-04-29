"""Evaluation metric aggregation utilities for multilingual runs."""

from __future__ import annotations

from collections import defaultdict

from .utils import MRR_at_5, recall_at_K


def _new_stage_buckets(
    fusion_top_k: int, recall_cutoffs: list[int]
) -> dict[str, dict[str, list[float]]]:
    """Create empty metric buckets for each pipeline stage.

    Args:
        fusion_top_k: Cutoff used for the RRF recall metric key.
        recall_cutoffs: Recall cutoffs to track for each stage.

    Returns:
        Nested dict of metric lists keyed by stage and metric name.
    """
    recall_keys = [f"r{k}" for k in recall_cutoffs]
    return {
        "dense": {m: [] for m in (["mrr5"] + recall_keys)},
        "sparse": {m: [] for m in (["mrr5"] + recall_keys)},
        "rrf": {m: [] for m in (["mrr5"] + recall_keys)},
        "final": {m: [] for m in (["mrr5"] + recall_keys)},
    }


class EvaluationMetrics:
    """Collect per-query metrics and produce language/global summaries."""

    def __init__(self, fusion_top_k: int = 30, recall_cutoffs: list[int] | None = None):
        """Initialize empty accumulators.

        Args:
            fusion_top_k: Cutoff used for RRF recall in reporting.
            recall_cutoffs: Recall cutoffs to track. Defaults to
                ``[3, 5] + list(range(10, 501, 10))``.
        """
        if recall_cutoffs is None:
            recall_cutoffs = [3, 5] + list(range(10, 501, 10))
        if not recall_cutoffs:
            raise ValueError("recall_cutoffs must not be empty")
        if any(k <= 0 for k in recall_cutoffs):
            raise ValueError("recall_cutoffs must be positive integers")

        self.fusion_top_k = fusion_top_k
        self.recall_cutoffs = sorted(set(int(k) for k in recall_cutoffs))
        self._by_language: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: _new_stage_buckets(self.fusion_top_k, self.recall_cutoffs)
        )
        self._counts: dict[str, int] = defaultdict(int)
        self._labeled_counts: dict[str, int] = defaultdict(int)

    def add_query(
        self,
        lang: str,
        true_pubkey: str | None,
        stages: dict[str, list[str]],
    ):
        """Add one query result to the metric accumulators.

        Args:
            lang: Language code for this query.
            true_pubkey: Ground-truth publication key. If missing, only count is updated.
            stages: Stage outputs keyed by ``dense``, ``sparse``, ``rrf``, and ``final``.
        """
        self._counts[lang] += 1
        if not true_pubkey:
            return
        self._labeled_counts[lang] += 1

        buckets = self._by_language[lang]
        for stage_name in ("dense", "sparse", "rrf", "final"):
            preds = stages.get(stage_name, [])
            buckets[stage_name]["mrr5"].append(MRR_at_5(preds, true_pubkey))
            for k in self.recall_cutoffs:
                buckets[stage_name][f"r{k}"].append(recall_at_K(preds, true_pubkey, k))

    @staticmethod
    def _safe_average(values: list[float]) -> float:
        """Return the average of values or ``0.0`` when empty."""
        return sum(values) / len(values) if values else 0.0

    def _summarize_language(self, lang: str) -> dict[str, object]:
        """Build a summary object for one language.

        Args:
            lang: Language code.

        Returns:
            Summary dict containing query counts and averaged metrics.
        """
        metrics = self._by_language.get(lang)
        total_queries = self._counts[lang]
        labeled_queries = self._labeled_counts[lang]
        unlabeled_queries = total_queries - labeled_queries
        if not metrics:
            return {
                "Total Queries": total_queries,
                "Labeled Queries": labeled_queries,
                "Unlabeled Queries": unlabeled_queries,
                "metrics": None,
            }

        summary = {}
        for stage_name, stage_metrics in metrics.items():
            summary[stage_name] = {
                metric_name: self._safe_average(values)
                for metric_name, values in stage_metrics.items()
            }
        return {
            "Total Queries": total_queries,
            "Labeled Queries": labeled_queries,
            "Unlabeled Queries": unlabeled_queries,
            "metrics": summary,
        }

    def summary(self) -> dict[str, object]:
        """Return language-level and global evaluation summaries."""
        language_summary = {
            lang: self._summarize_language(lang) for lang in sorted(self._counts)
        }

        global_totals = _new_stage_buckets(self.fusion_top_k, self.recall_cutoffs)
        total_queries = 0
        total_labeled_queries = 0
        for lang, data in language_summary.items():
            total_queries += data["Total Queries"]
            metrics = data["metrics"]
            total_labeled_queries += data["Labeled Queries"]
            if metrics is None:
                continue
            for stage_name, stage_metrics in metrics.items():
                for metric_name, value in stage_metrics.items():
                    global_totals[stage_name][metric_name].append(value)

        global_summary = None
        if total_labeled_queries > 0:
            global_summary = {
                stage_name: {
                    metric_name: self._safe_average(values)
                    for metric_name, values in stage_metrics.items()
                }
                for stage_name, stage_metrics in global_totals.items()
            }
        return {
            "languages": language_summary,
            "global": global_summary,
            "total_queries": total_queries,
            "total_labeled_queries": total_labeled_queries,
            "total_unlabeled_queries": total_queries - total_labeled_queries,
            "recall_cutoffs": self.recall_cutoffs,
        }
