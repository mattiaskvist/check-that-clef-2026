"""Evaluation helpers."""

from collections.abc import Sequence


def validate_prediction_shape(top5_preds: Sequence[Sequence[object]]) -> bool:
    if not top5_preds:
        return False
    return all(len(row) == 5 for row in top5_preds)
