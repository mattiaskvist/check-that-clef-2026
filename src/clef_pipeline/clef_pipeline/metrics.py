"""Evaluation metric aggregation utilities for multilingual runs."""

from __future__ import annotations

from collections import defaultdict

from .utils import MRR_at_5, recall_at_K


def _new_stage_buckets(fusion_top_k: int) -> dict[str, dict[str, list[float]]]:
    """Create empty metric buckets for each pipeline stage.

    Args:
        fusion_top_k: Cutoff used for the RRF recall metric key.

    Returns:
        Nested dict of metric lists keyed by stage and metric name.
    """
    return {
        "dense": {m: [] for m in ["mrr5", "r5", "r10", "r30", "r50"]},
        "sparse": {m: [] for m in ["mrr5", "r5", "r10", "r30", "r50"]},
        "rrf": {"mrr5": [], f"r{fusion_top_k}": []},
        "final": {"mrr5": [], "r5": []},
    }


class EvaluationMetrics:
    """Collect per-query metrics and produce language/global summaries."""

    def __init__(self, fusion_top_k: int = 30):
        """Initialize empty accumulators.

        Args:
            fusion_top_k: Cutoff used for RRF recall in reporting.
        """
        self.fusion_top_k = fusion_top_k
        self._by_language: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: _new_stage_buckets(self.fusion_top_k)
        )
        self._counts: dict[str, int] = defaultdict(int)

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

        buckets = self._by_language[lang]
        for stage_name in ("dense", "sparse"):
            preds = stages.get(stage_name, [])
            buckets[stage_name]["mrr5"].append(MRR_at_5(preds, true_pubkey))
            buckets[stage_name]["r5"].append(recall_at_K(preds, true_pubkey, 5))
            buckets[stage_name]["r10"].append(recall_at_K(preds, true_pubkey, 10))
            buckets[stage_name]["r30"].append(recall_at_K(preds, true_pubkey, 30))
            buckets[stage_name]["r50"].append(recall_at_K(preds, true_pubkey, 50))

        rrf_preds = stages.get("rrf", [])
        buckets["rrf"]["mrr5"].append(MRR_at_5(rrf_preds, true_pubkey))
        buckets["rrf"][f"r{self.fusion_top_k}"].append(
            recall_at_K(rrf_preds, true_pubkey, self.fusion_top_k)
        )

        final_preds = stages.get("final", [])
        buckets["final"]["mrr5"].append(MRR_at_5(final_preds, true_pubkey))
        buckets["final"]["r5"].append(recall_at_K(final_preds, true_pubkey, 5))

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
        if not metrics:
            return {"Total Queries": self._counts[lang], "metrics": None}

        summary = {}
        for stage_name, stage_metrics in metrics.items():
            summary[stage_name] = {
                metric_name: self._safe_average(values)
                for metric_name, values in stage_metrics.items()
            }
        return {"Total Queries": self._counts[lang], "metrics": summary}

    def summary(self) -> dict[str, object]:
        """Return language-level and global evaluation summaries."""
        language_summary = {
            lang: self._summarize_language(lang) for lang in sorted(self._counts)
        }

        global_totals = _new_stage_buckets(self.fusion_top_k)
        total_labeled_queries = 0
        for lang, data in language_summary.items():
            metrics = data["metrics"]
            if metrics is None:
                continue
            total_labeled_queries += data["Total Queries"]
            for stage_name, stage_metrics in metrics.items():
                for metric_name, value in stage_metrics.items():
                    global_totals[stage_name][metric_name].append(value)

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
            "total_labeled_queries": total_labeled_queries,
        }
