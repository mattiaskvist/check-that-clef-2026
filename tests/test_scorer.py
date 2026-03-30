import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scorer import scorer  # noqa: E402


def test_scorer_handles_string_predictions_against_integer_labels():
    top5_preds = [
        ["2764", "1", "2", "3", "4"],  # rank 1 hit
        ["0", "4854", "2", "3", "4"],  # rank 2 hit
    ]
    score = scorer(top5_preds, lang="en", split="dev")
    assert score > 0.0
