from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def index(self, corpus: list[str]):
        """Ingest the corpus and prepare the search index."""
        pass

    @abstractmethod
    def search(self, query: str) -> list[int]:
        """Return a sorted list of document indices based on relevance."""
        pass


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, doc_indices: list[int], corpus: list[str]
    ) -> list[tuple[int, float]]:
        """Return a sorted list of tuples (doc_index, score)."""
        pass


class BaseScorer(ABC):
    @abstractmethod
    def score(self, query: str, document: str) -> float:
        """Return a relevance score for a single (query, document) pair."""
        pass
