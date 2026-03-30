"""Gemini client wrappers for extraction and embeddings."""

from __future__ import annotations

from google import genai

from .config import RetrievalConfig
from .schemas import PaperEvidence, TweetEvidence


def build_embedding_input(evidence: TweetEvidence, raw_tweet: str) -> str:
    """Use structured evidence text when available, otherwise raw tweet text."""
    structured = evidence.query_text_for_embedding
    return structured if structured.strip() else raw_tweet


class GeminiService:
    """Thin wrapper around Gemini models used by retrieval pipeline."""

    def __init__(self, config: RetrievalConfig | None = None, client: genai.Client | None = None):
        self.config = config or RetrievalConfig()
        self.client = client or genai.Client()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.models.embed_content(
            model=self.config.embedding_model,
            contents=texts,
        )
        return [list(item.values) for item in response.embeddings]

    def extract_tweet_evidence(self, tweet_text: str) -> TweetEvidence:
        prompt = (
            "Extract retrieval evidence from the tweet. "
            "Return only JSON matching the schema."
            f"\n\nTweet:\n{tweet_text}"
        )
        response = self.client.models.generate_content(
            model=self.config.query_model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": TweetEvidence,
                "temperature": 0.0,
            },
        )
        return response.parsed

    def extract_paper_evidence(self, paper_prompt: str) -> PaperEvidence:
        response = self.client.models.generate_content(
            model=self.config.query_model,
            contents=paper_prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": PaperEvidence,
                "temperature": 0.0,
            },
        )
        return response.parsed
