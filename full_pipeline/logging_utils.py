import logging
from time import perf_counter


def get_logger(name: str = "full_pipeline") -> logging.Logger:
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
    def __init__(self):
        self._start = perf_counter()

    def elapsed_seconds(self) -> float:
        return perf_counter() - self._start
