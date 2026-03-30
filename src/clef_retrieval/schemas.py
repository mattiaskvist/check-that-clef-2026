"""Structured schemas for retrieval enrichment."""

from pydantic import BaseModel, Field


class TweetEvidence(BaseModel):
    claim_summary: str = ""
    language: str = "unknown"
    candidate_title_mentions: list[str] = Field(default_factory=list)
    candidate_authors: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    method_terms: list[str] = Field(default_factory=list)
    finding_terms: list[str] = Field(default_factory=list)
    domain_terms: list[str] = Field(default_factory=list)
    time_or_venue_hints: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    query_text_for_embedding: str


class PaperEvidence(BaseModel):
    pubkey: str
    title: str
    authors: str
    abstract: str
    venue: str = ""
    keywords: list[str] = Field(default_factory=list)
    method_terms: list[str] = Field(default_factory=list)
    finding_terms: list[str] = Field(default_factory=list)
    domain_terms: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    paper_text_for_embedding: str
