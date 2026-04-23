"""On-disk translation cache used by the sparse retriever.

Query-time translation was previously done per query with ``deep_translator``,
which is slow, rate-limited, and silently falls back on exceptions. This
module batches translations once per ``index_queries`` call and persists
results to a JSON file keyed by ``(source_lang, target_lang, sha256(text))``.
Subsequent runs with the same queries read from disk, making sparse indexing
deterministic and reproducible across evaluation runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Callable, Iterable


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _cache_key(source_lang: str, target_lang: str, text: str) -> str:
    return f"{source_lang}|{target_lang}|{_hash(text)}"


def _normalize_text(text: str) -> str:
    """Strip URLs, mentions, emoji, and collapse whitespace before translation."""
    text = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", text)
    text = re.sub(
        r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U000024C2-\U0001F251]+",
        " ",
        text,
    )
    text = text.replace("#", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TranslationCache:
    """Simple file-backed translation dictionary."""

    def __init__(self, path: str | None):
        """Create a translation cache.

        Args:
            path: Path to the JSON cache file. ``None`` disables persistence
                (in-memory only).
        """
        self.path = path
        self._entries: dict[str, str] = {}
        self._dirty = False
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    self._entries = json.load(handle)
            except (json.JSONDecodeError, OSError):
                self._entries = {}

    def get(self, source_lang: str, target_lang: str, text: str) -> str | None:
        return self._entries.get(_cache_key(source_lang, target_lang, text))

    def put(self, source_lang: str, target_lang: str, text: str, value: str) -> None:
        self._entries[_cache_key(source_lang, target_lang, text)] = value
        self._dirty = True

    def save(self) -> None:
        if not self.path or not self._dirty:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self._entries, handle, ensure_ascii=False)
        self._dirty = False


def translate_batch(
    texts: Iterable[str],
    source_lang: str,
    target_lang: str = "en",
    cache: TranslationCache | None = None,
    translator_factory: Callable[[str, str], object] | None = None,
) -> list[str]:
    """Translate every text once, reading/writing through ``cache``.

    Args:
        texts: Input texts to translate.
        source_lang: Source language code (``en``/``de``/``fr``/``auto``).
        target_lang: Target language code.
        cache: Optional persistent cache. When ``None``, every call re-translates.
        translator_factory: Callable returning an object exposing ``translate(text=...)``.
            Defaults to ``deep_translator.GoogleTranslator``. Tests can inject
            a stub to avoid network calls.

    Returns:
        Translated strings in the same order as ``texts``. On translator
        failure, falls back to the (normalized) source text for that entry.
    """
    texts_list = list(texts)
    if source_lang == target_lang:
        return [_normalize_text(t) or t for t in texts_list]

    if translator_factory is None:

        def _default_factory(source: str, target: str):
            from deep_translator import GoogleTranslator

            return GoogleTranslator(source=source, target=target)

        translator_factory = _default_factory

    translator = None
    results: list[str] = []
    for text in texts_list:
        normalized = _normalize_text(text)
        if not normalized:
            results.append(text)
            continue

        if cache is not None:
            cached = cache.get(source_lang, target_lang, normalized)
            if cached is not None:
                results.append(cached)
                continue

        if translator is None:
            translator = translator_factory(source_lang, target_lang)

        try:
            translated = translator.translate(text=normalized)
            if not translated:
                translated = normalized
        except Exception:
            translated = normalized

        results.append(translated)
        if cache is not None:
            cache.put(source_lang, target_lang, normalized, translated)

    if cache is not None:
        cache.save()

    return results
