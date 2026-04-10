from abc import ABC, abstractmethod

import numpy as np


class BaseRetriever(ABC):
    @abstractmethod
    def index(self, corpus: list[str]):
        """Ingest the corpus and prepare the search index."""
        pass

    @abstractmethod
    def search(self, query, **kwargs) -> tuple[list[int], np.ndarray]:
        """Return a sorted list of document indices and a scores array.

        Returns:
            tuple of (ranked_indices, scores) where:
                - ranked_indices: list of document indices sorted by relevance (descending)
                - scores: numpy array where scores[doc_id] = relevance score
        """
        pass


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, doc_indices: list[int], corpus: list[str]
    ) -> list[tuple[int, float]]:
        """Return a sorted list of tuples (doc_index, score)."""
        pass

