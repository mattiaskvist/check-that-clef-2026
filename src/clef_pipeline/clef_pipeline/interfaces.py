"""Abstract contracts for retrievers and rerankers."""

from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Interface for retrieval backends used by the pipeline."""

    @abstractmethod
    def index(self, corpus: list[str]):
        """Ingest the corpus and prepare the search index.

        Args:
            corpus: Preprocessed document texts to index.
        """
        pass

    @abstractmethod
    def search(self, query: str) -> list[int]:
        """Return ranked document indices for a query.

        Args:
            query: Query string or query identifier supported by the implementation.

        Returns:
            Ranked document indices in descending relevance order.
        """
        pass


class BaseReranker(ABC):
    """Base interface and helpers for reranking candidate documents."""

    @staticmethod
    def document_to_text(doc: str | dict) -> str:
        """Convert a document payload into reranker input text.

        Args:
            doc: Source document as raw text or metadata dictionary.

        Returns:
            Normalized text fed to reranker models.

        Raises:
            TypeError: If ``doc`` is neither ``str`` nor ``dict``.
        """
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
        """Convert a mixed corpus into plain text entries for reranking.

        Args:
            corpus: Documents represented as strings or dictionaries.

        Returns:
            Text-only representation of all documents.
        """
        return [self.document_to_text(doc) for doc in corpus]

    @abstractmethod
    def rerank(
        self, query: str, doc_indices: list[int], corpus: list[str]
    ) -> list[tuple[int, float]]:
        """Score and rank candidate document indices for a query.

        Args:
            query: Query text used for reranking.
            doc_indices: Candidate document indices from retrieval stages.
            corpus: Text corpus aligned with the candidate indices.

        Returns:
            Sorted ``(document_index, score)`` tuples in descending score order.
        """
        pass
