import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google.genai import errors
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clef_retrieval.config import RetrievalConfig  # noqa: E402
from clef_retrieval.data import load_collection, load_language_split  # noqa: E402
from clef_retrieval.paper_index import (  # noqa: E402
    IndexCacheError,
    build_or_load_index,
    index_paths,
    load_jsonl,
    validate_cached_index,
)
from clef_retrieval.pipeline import (  # noqa: E402
    build_query_embedding_text_with_metadata,
    rank_from_query_embedding,
)
from clef_retrieval.query_policy import (  # noqa: E402
    select_seeded_subset_indices,
    should_promote_from_subset,
)
from scorer import scorer  # noqa: E402


class MissingGeminiKeyError(RuntimeError):
    """Raised when GEMINI_API_KEY is required but missing."""


def _require_gemini_key() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise MissingGeminiKeyError(
            "Missing GEMINI_API_KEY. Set it in your environment or .env file before running this command."
        )


def _load_cached_index(config: RetrievalConfig) -> tuple[np.ndarray, list[dict]]:
    embeddings_path, metadata_path = index_paths(config)
    if not embeddings_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Index cache not found in {config.cache_dir}. Run `build-index` first."
        )
    embeddings = np.load(embeddings_path)
    metadata = load_jsonl(metadata_path)
    validate_cached_index(embeddings, metadata)
    return embeddings, metadata


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _predict_rows(
    config: RetrievalConfig,
    lang: str,
    split: str,
    limit: int | None,
    query_batch_size: int | None,
    skip_query_extraction: bool,
    subset_per_language_limit: int | None = None,
    subset_seed: int | None = None,
    include_extraction_outcomes: bool = False,
) -> list[dict[str, object]] | tuple[list[dict[str, object]], dict[str, int]]:
    from clef_retrieval.gemini_client import GeminiService

    _require_gemini_key()
    service = GeminiService(config=config)
    embeddings, metadata_rows = _load_cached_index(config)

    rows = list(load_language_split(lang, split))
    if limit is not None:
        rows = rows[:limit]
    rows, _ = _select_subset_rows(rows, subset_per_language_limit, subset_seed)
    batch_size = query_batch_size or config.query_batch_size

    predictions: list[dict[str, object]] = []
    extraction_outcomes = {"parsed+accepted": 0, "parsed+rejected": 0, "error->fallback": 0}
    for i in tqdm(
        range(0, len(rows), batch_size),
        desc=f"predict {lang}/{split}",
        unit="batch",
        disable=not sys.stderr.isatty(),
    ):
        batch_rows = rows[i : i + batch_size]
        if skip_query_extraction:
            query_texts = [str(row.get("text", "")) for row in batch_rows]
            extraction_outcomes["error->fallback"] += len(batch_rows)
        else:
            query_texts = []
            for row in batch_rows:
                query_text, outcome = _build_query_text_with_outcome(str(row.get("text", "")), service, config)
                query_texts.append(query_text)
                extraction_outcomes[outcome] += 1
        query_embeddings = service.embed_texts(query_texts)
        if len(query_embeddings) != len(batch_rows):
            raise ValueError("Query embedding count mismatch for prediction batch.")

        for row, query_embedding in zip(batch_rows, query_embeddings):
            top5 = rank_from_query_embedding(
                tweet_text=str(row.get("text", "")),
                query_embedding=query_embedding,
                paper_embeddings=embeddings,
                metadata_rows=metadata_rows,
                top_k=config.top_k,
            )
            predictions.append(
                {
                    "index": row.get("index"),
                    "text": row.get("text", ""),
                    "pubkey": row.get("pubkey"),
                    "top5": top5,
                }
            )
    if include_extraction_outcomes:
        return predictions, extraction_outcomes
    return predictions


def _build_query_text_with_outcome(
    tweet_text: str,
    service,
    config: RetrievalConfig,
) -> tuple[str, str]:
    query_text, outcome, _ = build_query_embedding_text_with_metadata(tweet_text, service, config)
    return query_text, outcome


def _select_subset_rows(
    rows: list[dict[str, object]],
    subset_limit: int | None,
    subset_seed: int | None,
) -> tuple[list[dict[str, object]], list[int]]:
    if subset_limit is None:
        return rows, list(range(len(rows)))
    seed = subset_seed if subset_seed is not None else 0
    indices = select_seeded_subset_indices(len(rows), subset_limit, seed)
    return [rows[idx] for idx in indices], indices


def _estimate_query_api_usage(query_count: int, skip_query_extraction: bool) -> dict[str, int | str | float]:
    extraction_calls = 0 if skip_query_extraction else query_count
    extraction_mode = "disabled" if skip_query_extraction else "enabled"
    calls_per_query = 1 if skip_query_extraction else 2
    embedding_calls = query_count
    total_calls = extraction_calls + embedding_calls
    return {
        "query_count": query_count,
        "extraction_mode": extraction_mode,
        "calls_per_query": calls_per_query,
        "extraction_calls": extraction_calls,
        "embedding_calls": embedding_calls,
        "total_calls": total_calls,
    }


def _enforce_cost_guardrail(total_calls: int, max_calls: int | None, allow_exceed: bool) -> str:
    if max_calls is None or total_calls <= max_calls:
        return "OK"
    if allow_exceed:
        return "WARNING"
    raise ValueError(f"Cost guardrail exceeded: estimated calls {total_calls} > limit {max_calls}")


def _predict_result_with_outcomes(
    result: list[dict[str, object]] | tuple[list[dict[str, object]], dict[str, int]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    if isinstance(result, tuple):
        rows, outcomes = result
        return rows, outcomes
    return result, {"parsed+accepted": 0, "parsed+rejected": 0, "error->fallback": len(result)}


def _default_prediction_path(config: RetrievalConfig, lang: str, split: str) -> Path:
    return Path(config.cache_dir) / f"predictions_{lang}_{split}.jsonl"


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _read_predictions(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                parsed = json.loads(stripped)
                if not isinstance(parsed, dict):
                    raise ValueError("Prediction row must be a JSON object.")
                rows.append(parsed)
    return rows


def _build_index(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    config = RetrievalConfig()
    embeddings_path, metadata_path = index_paths(config)

    if not args.force_rebuild and embeddings_path.exists() and metadata_path.exists():
        embeddings = np.load(embeddings_path)
        metadata = load_jsonl(metadata_path)
        validate_cached_index(embeddings, metadata)
        print(f"Using existing index cache in {config.cache_dir}")
        return 0

    _require_gemini_key()
    from clef_retrieval.gemini_client import GeminiService

    service = GeminiService(config=config)
    collection = [dict(row) for row in load_collection()]
    if args.limit_papers is not None:
        collection = collection[: args.limit_papers]
        if not collection:
            raise ValueError("--limit-papers must keep at least one collection row.")
    iterable = tqdm(
        collection,
        desc="build-index metadata",
        unit="paper",
        disable=not sys.stderr.isatty(),
    )
    result = build_or_load_index(
        collection=[dict(row) for row in iterable],
        service=service,
        config=config,
        force_rebuild=args.force_rebuild,
    )
    status = "built" if result["built"] else "loaded"
    print(
        f"Index {status}: {result['count']} papers -> {result['embeddings_path']} and {result['metadata_path']}"
    )
    elapsed = time.perf_counter() - start
    rate = result["count"] / elapsed if elapsed > 0 else 0.0
    print(f"Build-index summary: {result['count']} papers in {elapsed:.1f}s ({rate:.2f} papers/s)")
    return 0


def _predict(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    config = RetrievalConfig()
    query_model = "disabled (raw tweet text)" if args.skip_query_extraction else config.query_model
    print(f"Models: embedding={config.embedding_model}, query-extraction={query_model}")
    raw_rows = list(load_language_split(args.lang, args.split))
    if args.limit is not None:
        raw_rows = raw_rows[: args.limit]
    subset_limit = args.subset_per_language_limit
    subset_seed = args.subset_seed
    selected_rows, selected_indices = _select_subset_rows(raw_rows, subset_limit, subset_seed)
    usage = _estimate_query_api_usage(
        query_count=len(selected_rows),
        skip_query_extraction=args.skip_query_extraction,
    )
    projected_cost = float(usage["total_calls"]) * args.estimated_cost_per_call_sek
    guardrail_status = _enforce_cost_guardrail(
        total_calls=usage["total_calls"],
        max_calls=args.max_estimated_api_calls,
        allow_exceed=args.allow_cost_overrun,
    )
    print(
        "Usage estimate: "
        f"queries={usage['query_count']} extraction_mode={usage['extraction_mode']} "
        f"calls_per_query={usage['calls_per_query']} extraction_calls={usage['extraction_calls']} "
        f"embedding_calls={usage['embedding_calls']} total_calls={usage['total_calls']} "
        f"projected_cost_sek={projected_cost:.2f}"
    )
    print(f"Guardrail status: {guardrail_status}")
    if subset_limit is not None:
        print(
            "Subset mode: enabled "
            f"(lang={args.lang}, split={args.split}, selected={len(selected_rows)}, seed={subset_seed}, indices={selected_indices})"
        )

    prediction_result = _predict_rows(
        config,
        args.lang,
        args.split,
        args.limit,
        args.query_batch_size,
        args.skip_query_extraction,
        subset_per_language_limit=subset_limit,
        subset_seed=subset_seed,
        include_extraction_outcomes=True,
    )
    rows, extraction_outcomes = _predict_result_with_outcomes(prediction_result)
    total_queries = max(len(rows), 1)
    print(
        "Extraction outcomes: "
        f"parsed+accepted={extraction_outcomes['parsed+accepted']} "
        f"parsed+rejected={extraction_outcomes['parsed+rejected']} "
        f"error->fallback={extraction_outcomes['error->fallback']} "
        f"(accepted_rate={extraction_outcomes['parsed+accepted'] / total_queries:.2%})"
    )
    output_path = Path(args.output) if args.output else _default_prediction_path(config, args.lang, args.split)
    _write_predictions(output_path, rows)
    print(f"Wrote {len(rows)} predictions to {output_path}")
    elapsed = time.perf_counter() - start
    rate = len(rows) / elapsed if elapsed > 0 else 0.0
    print(f"Predict summary: {len(rows)} tweets in {elapsed:.1f}s ({rate:.2f} tweets/s)")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    config = RetrievalConfig()
    prediction_path = Path(args.predictions) if args.predictions else _default_prediction_path(config, args.lang, args.split)
    if prediction_path.exists() and not args.recompute:
        print(f"Using cached predictions from {prediction_path}")
        rows = _read_predictions(prediction_path)
        if args.limit is not None:
            rows = rows[: args.limit]
    else:
        query_model = "disabled (raw tweet text)" if args.skip_query_extraction else config.query_model
        print(f"Models: embedding={config.embedding_model}, query-extraction={query_model}")
        prediction_result = _predict_rows(
            config,
            args.lang,
            args.split,
            args.limit,
            args.query_batch_size,
            args.skip_query_extraction,
            subset_per_language_limit=args.subset_per_language_limit,
            subset_seed=args.subset_seed,
        )
        rows, _ = _predict_result_with_outcomes(prediction_result)

    if not rows:
        raise ValueError("No predictions available for evaluation.")
    top5_preds = [
        list(row["top5"])
        for row in tqdm(
            rows,
            desc="evaluate scoring",
            unit="tweet",
            disable=not sys.stderr.isatty(),
        )
    ]
    score = scorer(top5_preds, lang=args.lang, split=args.split)
    print(f"MRR@5: {score:.6f} ({len(rows)} queries)")
    if args.multilingual_metrics:
        per_language_scores: dict[str, float] = {}
        for language in ("en", "de", "fr"):
            lang_prediction_path = Path(args.predictions) if args.predictions else _default_prediction_path(config, language, args.split)
            if not lang_prediction_path.exists():
                continue
            lang_rows = _read_predictions(lang_prediction_path)
            if args.limit is not None:
                lang_rows = lang_rows[: args.limit]
            if not lang_rows:
                continue
            lang_top5 = [list(row["top5"]) for row in lang_rows]
            lang_score = scorer(lang_top5, lang=language, split=args.split)
            per_language_scores[language] = lang_score
            print(f"MRR@5 [{language}]: {lang_score:.6f} ({len(lang_rows)} queries)")
        if per_language_scores:
            overall = sum(per_language_scores.values()) / len(per_language_scores)
            print(f"MRR@5 overall (en,de,fr): {overall:.6f}")
            if (
                args.promotion_baseline_overall is not None
                and args.promotion_baseline_en is not None
                and args.promotion_baseline_de is not None
                and args.promotion_baseline_fr is not None
            ):
                promote = should_promote_from_subset(
                    baseline_overall=args.promotion_baseline_overall,
                    candidate_overall=overall,
                    baseline_by_language={
                        "en": args.promotion_baseline_en,
                        "de": args.promotion_baseline_de,
                        "fr": args.promotion_baseline_fr,
                    },
                    candidate_by_language=per_language_scores,
                )
                status = "PROMOTE" if promote else "BLOCKED"
                print(f"Promotion gate: {status}")
    elapsed = time.perf_counter() - start
    rate = len(rows) / elapsed if elapsed > 0 else 0.0
    print(f"Evaluate summary: {len(rows)} tweets in {elapsed:.1f}s ({rate:.2f} tweets/s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLEF retrieval pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_index = subparsers.add_parser("build-index", help="Build paper index artifacts")
    build_index.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Recompute embeddings even when cache artifacts already exist",
    )
    build_index.add_argument(
        "--limit-papers",
        type=_positive_int,
        default=None,
        help="Build index using only the first N collection papers (debugging/smoke runs)",
    )
    build_index.set_defaults(handler=_build_index)

    predict = subparsers.add_parser("predict", help="Predict top-5 pubkeys for tweets")
    predict.add_argument("--lang", choices=["de", "en", "fr"], required=True)
    predict.add_argument("--split", choices=["train", "dev"], required=True)
    predict.add_argument("--limit", type=_positive_int, default=None)
    predict.add_argument("--query-batch-size", type=_positive_int, default=None)
    predict.add_argument("--subset-seed", type=int, default=42)
    predict.add_argument("--subset-per-language-limit", type=_positive_int, default=None)
    predict.add_argument("--max-estimated-api-calls", type=_positive_int, default=200)
    predict.add_argument("--estimated-cost-per-call-sek", type=float, default=0.10)
    predict.add_argument("--allow-cost-overrun", action="store_true")
    predict.add_argument(
        "--skip-query-extraction",
        action="store_true",
        help="Skip LLM tweet-structure extraction and embed raw tweet text directly (faster).",
    )
    predict.add_argument(
        "--output",
        default=None,
        help="Path to write JSONL predictions (default: <cache_dir>/predictions_<lang>_<split>.jsonl)",
    )
    predict.set_defaults(handler=_predict)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions on a split")
    evaluate.add_argument("--lang", choices=["de", "en", "fr"], required=True)
    evaluate.add_argument("--split", choices=["train", "dev"], required=True)
    evaluate.add_argument("--limit", type=_positive_int, default=None)
    evaluate.add_argument("--subset-seed", type=int, default=42)
    evaluate.add_argument("--subset-per-language-limit", type=_positive_int, default=None)
    evaluate.add_argument(
        "--predictions",
        default=None,
        help="Path to an existing JSONL predictions file (default: <cache_dir>/predictions_<lang>_<split>.jsonl)",
    )
    evaluate.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore cached predictions and recompute with Gemini before scoring.",
    )
    evaluate.add_argument("--query-batch-size", type=_positive_int, default=None)
    evaluate.add_argument(
        "--skip-query-extraction",
        action="store_true",
        help="Skip LLM tweet-structure extraction and embed raw tweet text directly (faster).",
    )
    evaluate.add_argument("--multilingual-metrics", action="store_true")
    evaluate.add_argument("--promotion-baseline-overall", type=float, default=None)
    evaluate.add_argument("--promotion-baseline-en", type=float, default=None)
    evaluate.add_argument("--promotion-baseline-de", type=float, default=None)
    evaluate.add_argument("--promotion-baseline-fr", type=float, default=None)
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except MissingGeminiKeyError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, IndexCacheError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except (errors.APIError, errors.ClientError) as error:
        print(f"Gemini API error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
