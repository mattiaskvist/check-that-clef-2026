import re
import sys
import time
import random
import logging
from pathlib import Path
from collections import Counter

from datasets import load_dataset

# full_pipeline is a sibling package — add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.getLogger("country_converter").setLevel(logging.ERROR)


# ==========================================
# COVID / NOISE BLOCKLIST
# ==========================================
_COVID_BLOCKLIST = {
    "covid", "covid-19", "covid19", "covid 19",
    "sars-cov-2", "sars-cov2", "sarscov2", "sars cov 2",
    "coronavirus", "corona", "corona virus",
    "pandemic", "the pandemic",
}


def _filter_covid(entities: list[str]) -> list[str]:
    """Remove COVID-related entities that pollute NER results."""
    return [e for e in entities if e.lower().strip() not in _COVID_BLOCKLIST]


# ==========================================
# EXTRACTORS
# ==========================================

def _nlp():
    """Lazy-load spaCy, blocking torch to avoid CUDA DLL errors on Windows."""
    if not hasattr(_nlp, "_model"):
        import sys
        # Block torch from loading — thinc (spacy's backend) tries to import
        # torch which fails with OSError on Windows when CUDA DLLs are broken.
        # Setting sys.modules["torch"] = None makes Python raise ImportError
        # instead, which thinc handles gracefully (CPU-only mode).
        _torch_backup = sys.modules.get("torch", "__MISSING__")
        sys.modules["torch"] = None
        try:
            import spacy
            _nlp._model = spacy.load("en_core_web_sm")
        finally:
            # Restore original state so other code isn't affected
            if _torch_backup == "__MISSING__":
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = _torch_backup
    return _nlp._model


def extract_persons(text: str) -> list[str]:
    return _filter_covid([ent.text for ent in _nlp()(text).ents if ent.label_ == "PERSON"])


def extract_locations(text: str) -> list[str]:
    return _filter_covid([ent.text for ent in _nlp()(text).ents if ent.label_ in ("GPE", "LOC")])


def extract_orgs(text: str) -> list[str]:
    return _filter_covid([ent.text for ent in _nlp()(text).ents if ent.label_ == "ORG"])





# ==========================================
# ADD / REMOVE EXTRACTORS HERE
# Maps extractor name -> (function, matching strategy)
# Matching strategies: "fuzzy", "location", "exact"
# ==========================================
EXTRACTORS: dict[str, tuple] = {
    "persons":     (extract_persons,     "fuzzy"),
    "locations":   (extract_locations,   "location"),
    "orgs":        (extract_orgs,        "fuzzy"),
}


# ==========================================
# NORMALIZATION & MATCHING
# ==========================================

def _cc():
    """Lazy-load country converter."""
    if not hasattr(_cc, "_instance"):
        import country_converter as coco
        _cc._instance = coco.CountryConverter()
    return _cc._instance


def _normalize_location(entity: str) -> list[str]:
    """Expand a location entity to include aliases and parent country.

    e.g. 'USA' -> ['usa', 'united states']
         'Melbourne' -> ['melbourne', 'australia']
    """
    from geotext import GeoText

    terms = [entity.lower().strip()]

    country = _cc().convert(names=entity, to="name_short", not_found=None)
    if country:
        terms.append(country.lower().strip())

    geo = GeoText(entity.title())
    if geo.country_mentions:
        iso = list(geo.country_mentions.keys())[0]
        parent = _cc().convert(names=iso, to="name_short", not_found=None)
        if parent:
            terms.append(parent.lower().strip())

    return list(set(terms))


def _matches(entities: list[str], target: str, strategy: str, threshold: int = 80) -> bool:
    """Return True if any entity matches the target using the given strategy."""
    from rapidfuzz import fuzz

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

class HardIndicatorScorer():
    """Scores a (query, document) pair based on hard indicators extracted from the query.

    For each entity extracted from the query:
    - If the entity is present in the document, score +1.
    - If the entity is NOT present in the document, score -1 (penalize false positive).
    
    If no entities are extracted, the score is 0.
    """

    def score(self, query: str, document: str) -> float:
        total_score = 0.0
        
        for extractor_fn, strategy in EXTRACTORS.values():
            entities = extractor_fn(query)
            for entity in entities:
                if _matches([entity], document, strategy):
                    total_score += 1.0
                else:
                    total_score -= 1.0
                    
        return total_score


# ==========================================
# EVALUATION — DETAILED STATISTICS
# ==========================================

# Number of random wrong papers to check per tweet for false positive estimation
FP_SAMPLE_SIZE = 50


def _evaluate(tweets: list[dict], collection_dict: dict) -> dict:
    """Run all extractors over every tweet and collect detailed per-extractor
    statistics for reliability analysis, including false positive rates.

    Returns a detailed results dict with per-extractor and combined data.
    """
    start_time = time.time()
    # Build list of all paper keys for false positive sampling
    all_pubkeys = list(collection_dict.keys())

    # Per-extractor tracking
    extractor_stats = {
        name: {
            "extracted_count": 0,      # tweets where extractor found ≥1 entity
            "matched_count": 0,        # tweets where extracted entity matched the CORRECT paper
            "fp_matched_count": 0,     # tweets where entity matched a WRONG paper (false positive)
            "fp_total_checked": 0,     # total wrong papers checked (for rate calc)
            "fp_wrong_papers_matched": 0,  # total individual wrong papers that matched
            "total_entities": 0,       # total number of individual entities extracted
            "entity_value_counts": Counter(),  # most common extracted values
            "matched_entity_values": Counter(),  # most common values that actually matched
            "unmatched_entity_values": Counter(),  # most common values that didn't match
            "entity_counts_per_tweet": [],  # distribution of #entities per tweet (when >0)
        }
        for name in EXTRACTORS
    }

    # Combined tracking
    combined_extracted = 0
    combined_matched = 0
    combined_fp = 0  # tweets where ANY extractor matched a wrong paper

    # Per-tweet detail: how many extractor types fired & matched
    types_extracted_per_tweet = []
    types_matched_per_tweet = []

    n = len(tweets)
    for i, tweet in enumerate(tweets):
        if i % 500 == 0:
            print(f"Progress: {i}/{n} tweets", flush=True)
        paper = collection_dict.get(tweet["pubkey"])
        if not paper:
            continue

        paper_text = _paper_text(paper)
        tweet_extracted = False
        tweet_matched = False
        tweet_fp = False
        n_types_extracted = 0
        n_types_matched = 0

        # Sample random wrong papers for false positive checking
        wrong_keys = [k for k in random.sample(all_pubkeys, min(FP_SAMPLE_SIZE + 1, len(all_pubkeys)))
                      if k != tweet["pubkey"]][:FP_SAMPLE_SIZE]
        wrong_papers_text = [_paper_text(collection_dict[k]) for k in wrong_keys]

        for name, (extractor_fn, strategy) in EXTRACTORS.items():
            entities = extractor_fn(tweet["text"])
            if entities:
                stats = extractor_stats[name]
                stats["extracted_count"] += 1
                stats["total_entities"] += len(entities)
                stats["entity_counts_per_tweet"].append(len(entities))
                tweet_extracted = True
                n_types_extracted += 1

                # Track each entity value
                for ent in entities:
                    stats["entity_value_counts"][ent.lower().strip()] += 1

                matched_correct = _matches(entities, paper_text, strategy)

                if matched_correct:
                    stats["matched_count"] += 1
                    tweet_matched = True
                    n_types_matched += 1
                    for ent in entities:
                        stats["matched_entity_values"][ent.lower().strip()] += 1
                else:
                    for ent in entities:
                        stats["unmatched_entity_values"][ent.lower().strip()] += 1

                # False positive check: does this indicator match WRONG papers?
                n_wrong_matched = sum(
                    1 for wp_text in wrong_papers_text
                    if _matches(entities, wp_text, strategy)
                )
                stats["fp_total_checked"] += len(wrong_papers_text)
                stats["fp_wrong_papers_matched"] += n_wrong_matched

                if n_wrong_matched > 0:
                    stats["fp_matched_count"] += 1
                    tweet_fp = True

        types_extracted_per_tweet.append(n_types_extracted)
        types_matched_per_tweet.append(n_types_matched)

        if tweet_extracted:
            combined_extracted += 1
        if tweet_matched:
            combined_matched += 1
        if tweet_fp:
            combined_fp += 1

    return {
        "extractor_stats": extractor_stats,
        "combined_extracted": combined_extracted,
        "combined_matched": combined_matched,
        "combined_fp": combined_fp,
        "types_extracted_per_tweet": types_extracted_per_tweet,
        "types_matched_per_tweet": types_matched_per_tweet,
        "n_tweets": n,
        "execution_time_seconds": time.time() - start_time,
    }


def _print_results(results: dict):
    stats = results["extractor_stats"]
    n_tweets = results["n_tweets"]
    combined_extracted = results["combined_extracted"]
    combined_matched = results["combined_matched"]
    combined_fp = results["combined_fp"]
    types_extracted = results["types_extracted_per_tweet"]
    types_matched = results["types_matched_per_tweet"]

    # ──────────────────────────────────────
    # SECTION 1: Summary table
    # ──────────────────────────────────────
    col = 14
    print(f"\n{'='*100}")
    print(f"  HARD INDICATOR EXTRACTION — SUMMARY TABLE  (Processed {n_tweets} tweets in {results.get('execution_time_seconds', 0):.2f}s => {n_tweets/max(results.get('execution_time_seconds', 1), 0.001):.1f} queries/s)")
    print(f"{'='*100}")
    print(f"\n{'Extractor':<{col}} {'Extracted':>10} {'TP (correct)':>13} {'Precision':>10} {'Recall':>8} {'FP rate':>10} {'Selectivity':>12}")
    print("-" * 81)

    for name, s in stats.items():
        extracted = s["extracted_count"]
        matched = s["matched_count"]
        precision = (matched / extracted * 100) if extracted > 0 else 0
        recall = (matched / n_tweets * 100)
        # FP rate: fraction of wrong papers that matched (out of all wrong papers checked)
        fp_rate = (s["fp_wrong_papers_matched"] / s["fp_total_checked"] * 100) if s["fp_total_checked"] > 0 else 0
        # Selectivity: how much more likely to match correct vs random wrong paper
        # precision% / fp_rate% — higher is better
        selectivity = (precision / fp_rate) if fp_rate > 0 else float('inf') if precision > 0 else 0
        sel_str = f"{selectivity:.1f}x" if selectivity != float('inf') else "inf"
        print(f"{name:<{col}} {extracted:>6}/{n_tweets:<4} {matched:>8}     {precision:>8.1f}% {recall:>7.1f}% {fp_rate:>9.2f}% {sel_str:>11}")

    print("-" * 81)
    combined_precision = (combined_matched / combined_extracted * 100) if combined_extracted > 0 else 0
    combined_recall = (combined_matched / n_tweets * 100)
    combined_fp_rate = (combined_fp / combined_extracted * 100) if combined_extracted > 0 else 0
    print(f"{'COMBINED':<{col}} {combined_extracted:>6}/{n_tweets:<4} {combined_matched:>8}     {combined_precision:>8.1f}% {combined_recall:>7.1f}% {combined_fp_rate:>9.1f}%")

    # ──────────────────────────────────────
    # SECTION 2: Per-extractor reliability deep-dive
    # ──────────────────────────────────────
    print(f"\n{'='*100}")
    print("  PER-EXTRACTOR RELIABILITY ANALYSIS")
    print(f"{'='*100}")

    for name, s in stats.items():
        extracted = s["extracted_count"]
        matched = s["matched_count"]
        if extracted == 0:
            print(f"\n── {name.upper()} ── never extracted any entities, skipping.")
            continue

        precision = matched / extracted * 100
        recall = matched / n_tweets * 100
        total_ents = s["total_entities"]
        avg_per_tweet = total_ents / extracted if extracted > 0 else 0
        counts = s["entity_counts_per_tweet"]
        min_count = min(counts) if counts else 0
        max_count = max(counts) if counts else 0
        fp_rate = (s["fp_wrong_papers_matched"] / s["fp_total_checked"] * 100) if s["fp_total_checked"] > 0 else 0
        fp_tweet_rate = (s["fp_matched_count"] / extracted * 100) if extracted > 0 else 0

        print(f"\n── {name.upper()} ──")
        print(f"  Tweets with extraction:  {extracted:>6} / {n_tweets}  ({extracted/n_tweets*100:.1f}%)")
        print(f"  Tweets matched correct:  {matched:>6} / {extracted}  (precision: {precision:.1f}%)")
        print(f"  Recall (matched/total):  {matched:>6} / {n_tweets}  ({recall:.1f}%)")
        print(f"  Total entities found:    {total_ents:>6}")
        print(f"  Avg entities per tweet:  {avg_per_tweet:>6.1f}  (min={min_count}, max={max_count})")
        print(f"  FALSE POSITIVE ANALYSIS:")
        print(f"    Tweets matching ≥1 wrong paper:  {s['fp_matched_count']:>6} / {extracted}  ({fp_tweet_rate:.1f}%)")
        print(f"    Wrong papers matched (of {s['fp_total_checked']}): {s['fp_wrong_papers_matched']}  ({fp_rate:.2f}%)")

        # Unique entities
        unique_extracted = len(s["entity_value_counts"])
        unique_matched = len(s["matched_entity_values"])
        print(f"  Unique extracted values: {unique_extracted:>6}")
        print(f"  Unique matched values:   {unique_matched:>6}")

        # Top extracted values
        top_n = 10
        print(f"\n  Top {top_n} most extracted values:")
        for val, cnt in s["entity_value_counts"].most_common(top_n):
            matched_cnt = s["matched_entity_values"].get(val, 0)
            ent_precision = matched_cnt / cnt * 100 if cnt > 0 else 0
            marker = "+" if matched_cnt > 0 else "-"
            print(f"    {marker} {val!r:30s}  count={cnt:>4}  matched={matched_cnt:>4}  ({ent_precision:.0f}%)")

        # Top matched values (that actually help)
        top_matched = s["matched_entity_values"].most_common(top_n)
        if top_matched:
            print(f"\n  Top {top_n} most reliable values (highest match count):")
            for val, cnt in top_matched:
                total_cnt = s["entity_value_counts"].get(val, cnt)
                ent_precision = cnt / total_cnt * 100 if total_cnt > 0 else 0
                print(f"    + {val!r:30s}  matched={cnt:>4}/{total_cnt:<4}  ({ent_precision:.0f}%)")

        # Top unmatched values (noise)
        top_unmatched = s["unmatched_entity_values"].most_common(5)
        if top_unmatched:
            print(f"\n  Top 5 noisiest values (extracted but never/rarely matched):")
            for val, cnt in top_unmatched:
                matched_cnt = s["matched_entity_values"].get(val, 0)
                print(f"    - {val!r:30s}  unmatched={cnt:>4}  matched={matched_cnt:>4}")

    # ──────────────────────────────────────
    # SECTION 3: False positive summary
    # ──────────────────────────────────────
    print(f"\n{'='*100}")
    print("  FALSE POSITIVE SUMMARY")
    print(f"  (for each tweet with extracted entities, checked against {FP_SAMPLE_SIZE} random wrong papers)")
    print(f"{'='*100}")

    print(f"\n  {'Extractor':<14} {'Extracted':>10} {'TP correct':>11} {'FP any wrong':>13} {'TP only':>10} {'Discrimination':>15}")
    print(f"  {'-'*77}")

    for name, s in stats.items():
        extracted = s["extracted_count"]
        matched = s["matched_count"]
        fp_tweets = s["fp_matched_count"]
        # "TP only" = matched correct AND did NOT match any wrong paper
        # We don't track this exactly at tweet level, but we can show the rates
        tp_rate = (matched / extracted * 100) if extracted > 0 else 0
        fp_tweet_rate = (fp_tweets / extracted * 100) if extracted > 0 else 0
        # Discrimination = how much better than random
        fp_per_paper = (s["fp_wrong_papers_matched"] / s["fp_total_checked"]) if s["fp_total_checked"] > 0 else 0
        tp_per_paper = (matched / extracted) if extracted > 0 else 0
        disc = tp_per_paper / fp_per_paper if fp_per_paper > 0 else float('inf') if tp_per_paper > 0 else 0
        disc_str = f"{disc:.1f}x" if disc != float('inf') else "inf"
        print(f"  {name:<14} {extracted:>10} {matched:>8} ({tp_rate:>4.0f}%) {fp_tweets:>8} ({fp_tweet_rate:>4.0f}%) {disc_str:>14}")

    print()
    print(f"  Interpretation:")
    print(f"    TP correct  = indicator found AND matches the correct paper")
    print(f"    FP any wrong = indicator found AND matches at least 1 of {FP_SAMPLE_SIZE} random wrong papers")
    print(f"    Discrimination = TP rate / FP-per-paper rate (higher = more useful for retrieval)")

    # ──────────────────────────────────────
    # SECTION 4: Coverage analysis — how many types fire per tweet
    # ──────────────────────────────────────
    print(f"\n{'='*100}")
    print("  COVERAGE ANALYSIS — EXTRACTOR TYPES PER TWEET")
    print(f"{'='*100}")

    extracted_counter = Counter(types_extracted)
    matched_counter = Counter(types_matched)

    print(f"\n  Distribution: how many extractor types found >=1 entity per tweet")
    print(f"  {'# types':>8}  {'tweets':>8}  {'% of total':>10}")
    for k in range(max(extracted_counter.keys()) + 1):
        cnt = extracted_counter.get(k, 0)
        print(f"  {k:>8}  {cnt:>8}  {cnt/n_tweets*100:>9.1f}%")

    print(f"\n  Distribution: how many extractor types MATCHED per tweet")
    print(f"  {'# types':>8}  {'tweets':>8}  {'% of total':>10}")
    for k in range(max(matched_counter.keys()) + 1):
        cnt = matched_counter.get(k, 0)
        print(f"  {k:>8}  {cnt:>8}  {cnt/n_tweets*100:>9.1f}%")

    avg_extracted = sum(types_extracted) / n_tweets
    avg_matched = sum(types_matched) / n_tweets
    print(f"\n  Avg extractor types firing per tweet:  {avg_extracted:.2f}")
    print(f"  Avg extractor types matching per tweet: {avg_matched:.2f}")
    print(f"{'='*100}\n")

    # Machine-readable metric line for autoresearch
    metric_value = combined_matched / n_tweets
    print(f"METRIC: {metric_value:.6f}")


def _log_to_tsv(results: dict, filename: str = "benchmark_results.tsv", description: str = "Baseline"):
    import csv
    import os
    from datetime import datetime

    n_tweets = results["n_tweets"]
    exec_time = results.get("execution_time_seconds", 0)
    qps = n_tweets / max(exec_time, 0.001)

    combined_extracted = results["combined_extracted"]
    combined_matched = results["combined_matched"]
    combined_fp = results["combined_fp"]

    precision = (combined_matched / combined_extracted * 100) if combined_extracted > 0 else 0
    recall = (combined_matched / n_tweets * 100)
    fp_rate = (combined_fp / combined_extracted * 100) if combined_extracted > 0 else 0

    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["Timestamp", "Description", "Exec Time (s)", "Queries/s", "Precision (%)", "Recall (%)", "FP Rate (%)"])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            timestamp,
            description,
            f"{exec_time:.2f}",
            f"{qps:.1f}",
            f"{precision:.2f}",
            f"{recall:.2f}",
            f"{fp_rate:.2f}"
        ])
    print(f"\n=> Appended benchmark metrics to {filename}")


def main():
    import sys
    desc = sys.argv[1] if len(sys.argv) > 1 else "Baseline"
    
    random.seed(42)  # reproducible false positive sampling
    print("Loading datasets...")
    collection_raw = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "collection"
    )["collection"]
    data_raw = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "en"
    )["train"]

    collection_dict = {row["pubkey"]: row for row in collection_raw.to_list()}
    tweets = data_raw.to_list()

    results = _evaluate(tweets, collection_dict)
    _print_results(results)
    _log_to_tsv(results, description=desc)

if __name__ == "__main__":
    main()