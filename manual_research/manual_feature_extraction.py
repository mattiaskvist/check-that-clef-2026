import re
import sys
import logging
from pathlib import Path

import spacy
import country_converter as coco
from geotext import GeoText
from rapidfuzz import fuzz

from datasets import load_dataset

# full_pipeline is a sibling package — add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from full_pipeline.interfaces import BaseScorer

logging.getLogger("country_converter").setLevel(logging.ERROR)


# ==========================================
# EXTRACTORS
# ==========================================

def _nlp():
    """Lazy-load spaCy to avoid requiring the model at import time."""
    if not hasattr(_nlp, "_model"):
        _nlp._model = spacy.load("en_core_web_sm")
    return _nlp._model


def extract_persons(text: str) -> list[str]:
    return [ent.text for ent in _nlp()(text).ents if ent.label_ == "PERSON"]


def extract_locations(text: str) -> list[str]:
    return [ent.text for ent in _nlp()(text).ents if ent.label_ in ("GPE", "LOC")]


def extract_orgs(text: str) -> list[str]:
    return [ent.text for ent in _nlp()(text).ents if ent.label_ == "ORG"]


def extract_events(text: str) -> list[str]:
    """Conference / event mentions e.g. NeurIPS, ICLR, ICML."""
    return [ent.text for ent in _nlp()(text).ents if ent.label_ == "EVENT"]


def extract_products(text: str) -> list[str]:
    """Product / model name mentions e.g. GPT-4, ResNet, ChatGPT."""
    return [ent.text for ent in _nlp()(text).ents if ent.label_ == "PRODUCT"]


def extract_statistics(text: str) -> list[str]:
    """Percentage figures e.g. '94.3%'."""
    return re.findall(r"\d+\.?\d*\s*%", text)


def extract_years(text: str) -> list[str]:
    """Plausible publication years (2000–2026)."""
    return re.findall(r"\b20(?:0[0-9]|1[0-9]|2[0-6])\b", text)


def extract_sample_sizes(text: str) -> list[str]:
    """Sample size patterns: 'n=1000', 'N = 500', '1234 participants'."""
    patterns = [
        r"\b[nN]\s*=\s*\d[\d,]+",
        r"\b\d[\d,]+\s+(?:participants|subjects|patients|samples|observations)\b",
    ]
    results = []
    for p in patterns:
        results.extend(re.findall(p, text, flags=re.IGNORECASE))
    return results


def extract_p_values(text: str) -> list[str]:
    """Statistical p-values: 'p < 0.05', 'p = 0.001'."""
    return re.findall(r"\bp[\s-]?(?:value\s*)?[<>=]\s*0\.\d+", text, flags=re.IGNORECASE)


def extract_acronyms(text: str) -> list[str]:
    """All-caps acronyms (2–5 letters) that are likely domain-specific.

    Common English words (I, A, US, TV, etc.) and years are filtered out.
    """
    _COMMON = {"I", "A", "US", "UK", "EU", "UN", "TV", "AM", "PM", "OK", "ID"}
    candidates = re.findall(r"\b[A-Z]{2,5}\b", text)
    return [c for c in candidates if c not in _COMMON and not c.isdigit()]


# ==========================================
# ADD / REMOVE EXTRACTORS HERE
# Maps extractor name -> (function, matching strategy)
# Matching strategies: "fuzzy", "location", "exact", "substring"
# ==========================================
EXTRACTORS: dict[str, tuple] = {
    "persons":      (extract_persons,      "fuzzy"),
    "locations":    (extract_locations,    "location"),
    "orgs":         (extract_orgs,         "fuzzy"),
    "events":       (extract_events,       "fuzzy"),
    "products":     (extract_products,     "fuzzy"),
    "statistics":   (extract_statistics,   "exact"),
    "years":        (extract_years,        "exact"),
    "sample_sizes": (extract_sample_sizes, "exact"),
    "p_values":     (extract_p_values,     "exact"),
    "acronyms":     (extract_acronyms,     "exact"),
}


# ==========================================
# NORMALIZATION & MATCHING
# ==========================================

_cc = coco.CountryConverter()


def _normalize_location(entity: str) -> list[str]:
    """Expand a location entity to include aliases and parent country.

    e.g. 'USA' -> ['usa', 'united states']
         'Melbourne' -> ['melbourne', 'australia']
    """
    terms = [entity.lower().strip()]

    country = _cc.convert(names=entity, to="name_short", not_found=None)
    if country:
        terms.append(country.lower().strip())

    geo = GeoText(entity.title())
    if geo.country_mentions:
        iso = list(geo.country_mentions.keys())[0]
        parent = _cc.convert(names=iso, to="name_short", not_found=None)
        if parent:
            terms.append(parent.lower().strip())

    return list(set(terms))


def _matches(entities: list[str], target: str, strategy: str, threshold: int = 80) -> bool:
    """Return True if any entity matches the target using the given strategy."""
    if not entities or not target:
        return False

    target_lower = target.lower()

    for entity in entities:
        if strategy == "exact":
            if entity.lower().strip() in target_lower:
                return True

        elif strategy == "location":
            for term in _normalize_location(entity):
                if fuzz.token_set_ratio(term, target_lower) >= threshold:
                    return True

        else:  # fuzzy
            if fuzz.token_set_ratio(entity.lower().strip(), target_lower) >= threshold:
                return True

    return False


def _paper_text(paper: dict) -> str:
    return " ".join([
        str(paper["title"]),
        str(paper["abstract"]),
        str(paper["authors"]),
        str(paper["venue"]),
    ])


# ==========================================
# SCORER (pipeline-compatible)
# ==========================================

class HardIndicatorScorer(BaseScorer):
    """Scores a (query, document) pair by counting how many hard indicator
    types extracted from the query are present in the document.

    Returns a float in [0, len(EXTRACTORS)].
    """

    def score(self, query: str, document: str) -> float:
        return float(sum(
            1
            for extractor_fn, strategy in EXTRACTORS.values()
            if _matches(extractor_fn(query), document, strategy)
        ))


# ==========================================
# EVALUATION
# ==========================================

def _evaluate(tweets: list[dict], collection_dict: dict) -> tuple[dict, int, int]:
    """Run all extractors over every tweet and measure match rates against
    the ground-truth paper.

    Returns per-extractor metrics and combined extracted/matched counts
    (each tweet counted only once in the combined totals).
    """
    metrics = {
        name: {"extracted_count": 0, "matched_in_paper": 0}
        for name in EXTRACTORS
    }
    combined_extracted = 0
    combined_matched   = 0

    n = len(tweets)
    for i, tweet in enumerate(tweets):
        if i % 1000 == 0:
            print(f"Progress: {i}/{n} tweets", flush=True)
        paper = collection_dict.get(tweet["pubkey"])
        if not paper:
            continue

        paper_text = _paper_text(paper)
        tweet_extracted = False
        tweet_matched   = False

        for name, (extractor_fn, strategy) in EXTRACTORS.items():
            entities = extractor_fn(tweet["text"])
            if entities:
                metrics[name]["extracted_count"] += 1
                tweet_extracted = True
                if _matches(entities, paper_text, strategy):
                    metrics[name]["matched_in_paper"] += 1
                    tweet_matched = True

        if tweet_extracted:
            combined_extracted += 1
        if tweet_matched:
            combined_matched += 1

    return metrics, combined_extracted, combined_matched


def _print_results(metrics: dict, combined_extracted: int, combined_matched: int, n_tweets: int):
    col = 16
    print(f"\n{'Extractor':<{col}} {'Tweets w/ entity':>16} {'Matched':>8} {'Match rate':>11} {'Overall%':>9}")
    print("-" * (col + 48))

    for name, stats in metrics.items():
        extracted = stats["extracted_count"]
        matched   = stats["matched_in_paper"]
        rate      = (matched / extracted * 100) if extracted > 0 else 0
        overall   = (matched / n_tweets * 100)
        print(f"{name:<{col}} {extracted:>12}/{n_tweets:<3} {matched:>8} {rate:>10.1f}% {overall:>8.1f}%")

    print("-" * (col + 48))
    combined_rate = (combined_matched / combined_extracted * 100) if combined_extracted > 0 else 0
    overall_rate  = (combined_matched / n_tweets * 100)
    print(f"{'COMBINED (any)':<{col}} {combined_extracted:>12}/{n_tweets:<3} {combined_matched:>8} {combined_rate:>10.1f}% {overall_rate:>8.1f}%")
    print("=" * (col + 48))

    # Machine-readable metric line for autoresearch
    metric_value = combined_matched / n_tweets
    print(f"METRIC: {metric_value:.6f}")


def main():
    print("Loading datasets...")
    collection_raw = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "collection"
    )["collection"]
    data_raw = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "en"
    )["train"]

    collection_dict = {row["pubkey"]: row for row in collection_raw.to_list()}
    tweets = data_raw.to_list()

    metrics, combined_extracted, combined_matched = _evaluate(tweets, collection_dict)
    _print_results(metrics, combined_extracted, combined_matched, len(tweets))


if __name__ == "__main__":
    main()