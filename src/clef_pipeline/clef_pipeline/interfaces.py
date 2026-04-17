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
    @staticmethod
    def document_to_text(doc: str | dict) -> str:
        if isinstance(doc, str):
            return doc.strip()

        if not isinstance(doc, dict):
            raise TypeError("Documents must be either strings or dictionaries.")

        title = str(doc.get("title") or "").strip()
        abstract = str(doc.get("abstract") or "").strip()
        venue = str(doc.get("venue") or "").strip()
        authors_raw = doc.get("authors") or ""
        if isinstance(authors_raw, list):
            authors = ", ".join(
                str(author).strip() for author in authors_raw if str(author).strip()
            )
        else:
            authors = str(authors_raw).strip()

        text = f"Title: {title} | Abstract: {abstract}"
        if venue:
            text += f" | Venue: {venue}"
        if authors:
            text += f" | Authors: {authors}"

        return text.strip()

    def preprocess_corpus(self, corpus: list[str | dict]) -> list[str]:
        return [self.document_to_text(doc) for doc in corpus]

    @abstractmethod
    def rerank(
        self, query: str, doc_indices: list[int], corpus: list[str]
    ) -> list[tuple[int, float]]:
        """Return a sorted list of tuples (doc_index, score)."""
        pass
