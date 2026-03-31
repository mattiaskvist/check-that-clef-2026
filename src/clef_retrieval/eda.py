"""Quick exploratory data analysis helpers."""

from statistics import mean, median


def summarize_lengths(values: list[str]) -> dict[str, float]:
    lengths = [len((value or "").split()) for value in values]
    if not lengths:
        return {"mean": 0.0, "median": 0.0}
    return {"mean": float(mean(lengths)), "median": float(median(lengths))}
