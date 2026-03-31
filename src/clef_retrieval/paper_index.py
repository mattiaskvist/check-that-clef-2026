"""Paper indexing helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .config import RetrievalConfig
from .data import normalize_authors
from .gemini_client import GeminiService
from .schemas import PaperEvidence

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


class IndexBuildError(RuntimeError):
    """Raised when index artifacts cannot be built."""


class IndexCacheError(RuntimeError):
    """Raised when cached index artifacts are inconsistent."""


def _extract_keywords(text: str, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for token in _TOKEN_PATTERN.findall((text or "").lower()):
        if len(token) <= 3 or token in seen:
            continue
        seen.add(token)
        values.append(token)
        if len(values) >= limit:
            break
    return values


def build_paper_text(evidence: PaperEvidence) -> str:
    parts = [
        f"Title: {evidence.title}",
        f"Authors: {evidence.authors}",
        f"Abstract: {evidence.abstract}",
    ]
    if evidence.venue:
        parts.append(f"Venue: {evidence.venue}")
    if evidence.keywords:
        parts.append(f"Keywords: {', '.join(evidence.keywords)}")
    if evidence.highlights:
        parts.append(f"Highlights: {'; '.join(evidence.highlights)}")
    return "\n".join(parts)


def save_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _build_metadata_row(paper: dict[str, Any]) -> dict[str, Any]:
    title = str(paper.get("title", "") or "")
    abstract = str(paper.get("abstract", "") or "")
    venue = str(paper.get("venue", "") or "")
    authors = normalize_authors(paper.get("authors"))
    keywords = _extract_keywords(f"{title} {abstract}")
    highlights = [title] if title else []

    evidence = PaperEvidence(
        pubkey=str(paper.get("pubkey", "")),
        title=title,
        authors=authors,
        abstract=abstract,
        venue=venue,
        keywords=keywords,
        highlights=highlights,
        paper_text_for_embedding="",
    )
    row = evidence.model_dump()
    row["paper_text_for_embedding"] = build_paper_text(evidence)
    return row


def index_paths(config: RetrievalConfig) -> tuple[Path, Path]:
    cache_dir = Path(config.cache_dir)
    return cache_dir / "paper_embeddings.npy", cache_dir / "paper_metadata.jsonl"


def build_or_load_index(
    collection: list[dict[str, Any]],
    service: GeminiService,
    config: RetrievalConfig,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    embeddings_path, metadata_path = index_paths(config)
    if not force_rebuild and embeddings_path.exists() and metadata_path.exists():
        metadata = load_jsonl(metadata_path)
        embeddings = np.load(embeddings_path)
        validate_cached_index(embeddings, metadata)
        return {
            "built": False,
            "embeddings_path": str(embeddings_path),
            "metadata_path": str(metadata_path),
            "count": len(metadata),
            "embeddings": embeddings,
            "metadata": metadata,
        }

    metadata_rows = [_build_metadata_row(dict(row)) for row in collection]
    embedding_inputs = [row["paper_text_for_embedding"] for row in metadata_rows]
    vectors = service.embed_texts(embedding_inputs)
    if len(vectors) != len(metadata_rows):
        raise IndexBuildError(
            "Embedding response count mismatch while building index artifacts."
        )

    matrix = np.asarray(vectors, dtype=float)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, matrix)
    save_jsonl(metadata_rows, metadata_path)
    return {
        "built": True,
        "embeddings_path": str(embeddings_path),
        "metadata_path": str(metadata_path),
        "count": len(metadata_rows),
        "embeddings": matrix,
        "metadata": metadata_rows,
    }


def validate_cached_index(embeddings: np.ndarray, metadata_rows: list[dict[str, Any]]) -> None:
    if embeddings.ndim != 2:
        raise IndexCacheError("Cached embeddings must be a 2D matrix.")
    if len(metadata_rows) == 0:
        raise IndexCacheError("Cached metadata is empty. Rebuild index cache.")
    if embeddings.shape[0] != len(metadata_rows):
        raise IndexCacheError(
            "Cached index is inconsistent: number of embedding rows does not match metadata rows. "
            "Run `build-index --force-rebuild`."
        )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows
