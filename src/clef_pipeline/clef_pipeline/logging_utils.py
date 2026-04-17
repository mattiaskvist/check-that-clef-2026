"""Logging and lightweight timing helpers for pipeline execution."""

import logging
from time import perf_counter


def get_logger(name: str = "clef_pipeline") -> logging.Logger:
    """Create or reuse a configured logger for CLI and Modal runs.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class StageTimer:
    """Monotonic wall-clock timer for coarse stage duration measurements."""

    def __init__(self):
        """Start the timer at construction time."""
        self._start = perf_counter()

    def elapsed_seconds(self) -> float:
        """Return elapsed time in seconds since timer creation."""
        return perf_counter() - self._start
