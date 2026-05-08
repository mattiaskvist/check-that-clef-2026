"""Provider-neutral Harrier hard-negative mining and LoRA fine-tuning.

This module keeps the heavy training implementation free of Modal APIs so it can
run on Google Cloud, local GPU machines, or any other container scheduler. The
default CLI target is German training with four GPU processes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable


CHECKTHAT_DATASET = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"
HARRIER_MODEL_NAME = "microsoft/harrier-oss-v1-27b"
DATA_MOUNT = "/data"
DEFAULT_LANGUAGE = "de"

LANGUAGE_ALIASES = {
    "en": "en",
    "english": "en",
    "de": "de",
    "german": "de",
    "fr": "fr",
    "french": "fr",
}

LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
}

LANGUAGE_QUERY_PROMPTS = {
    "en": (
        "Instruct: Retrieve the implicitly referenced medical-scientific "
        "publication for this English claim. Prioritize diseases, "
        "interventions, populations, measurements, acronyms, study names, "
        "and exact biomedical terms.\nQuery: "
    ),
    "de": (
        "Instruct: Retrieve the implicitly referenced medical-scientific "
        "publication for this German claim. Prioritize diseases, "
        "interventions, populations, measurements, acronyms, study names, "
        "and exact biomedical terms, even when the claim is written in "
        "German.\nQuery: "
    ),
    "fr": (
        "Instruct: Retrieve the implicitly referenced medical-scientific "
        "publication for this French claim. Prioritize diseases, "
        "interventions, populations, measurements, acronyms, study names, "
        "and exact biomedical terms, even when the claim is written in "
        "French.\nQuery: "
    ),
}


def normalize_language(language: str) -> str:
    """Normalize a language code or English language name to dataset config."""
    normalized = language.strip().lower()
    try:
        return LANGUAGE_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(LANGUAGE_ALIASES))
        raise ValueError(
            f"Unsupported language '{language}'. Use one of: {supported}"
        ) from exc


def language_name(language: str) -> str:
    """Return a display name for a normalized or alias language."""
    return LANGUAGE_NAMES[normalize_language(language)]


def query_prompt_for_language(language: str) -> str:
    """Return the Harrier query prompt for a language."""
    return LANGUAGE_QUERY_PROMPTS[normalize_language(language)]


def default_language_paths(
    data_dir: str = DATA_MOUNT,
    language: str = DEFAULT_LANGUAGE,
) -> dict[str, str]:
    """Build the default durable paths for one language training run."""
    lang = normalize_language(language)
    base = Path(data_dir).expanduser()
    return {
        "triplets_path": str(base / f"harrier_{lang}_hard_negatives.jsonl"),
        "document_embeddings_path": str(
            base / f"harrier_{lang}_document_embeddings.pt"
        ),
        "checkpoint_dir": str(base / f"harrier-27b-{lang}-checkpoints"),
        "output_dir": str(base / f"harrier-27b-{lang}-lora"),
        "distributed_args_path": str(base / f"harrier_{lang}_train_args.json"),
    }


def _ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def article_to_text(doc: dict) -> str:
    """Convert a collection document into dense-retriever training text."""
    title = str(doc.get("title") or "").strip()
    abstract = str(doc.get("abstract") or "").strip()
    return f"{title}\n{abstract}".strip()


def parse_target_modules(target_modules: str) -> str | list[str]:
    """Parse a PEFT target module setting from the CLI."""
    normalized = target_modules.strip()
    if normalized == "all-linear":
        return normalized
    parsed = [module.strip() for module in normalized.split(",") if module.strip()]
    if not parsed:
        raise ValueError("target_modules must be 'all-linear' or a comma list.")
    return parsed


def make_triplet_record(
    *,
    query_row: dict,
    positive_text: str,
    negative_text: str,
    negative_pubkey: str,
    negative_rank: int,
    language: str = DEFAULT_LANGUAGE,
    query_key: str | None = None,
) -> dict[str, object]:
    """Build one JSONL training record with metadata for auditing."""
    lang = normalize_language(language)
    resolved_query_key = query_key or str(query_row.get("index") or query_row["pubkey"])
    return {
        "query_key": resolved_query_key,
        "anchor": str(query_row["text"]).strip(),
        "positive": positive_text,
        "negative": negative_text,
        "query_pubkey": str(query_row["pubkey"]),
        "negative_pubkey": str(negative_pubkey),
        "negative_rank": int(negative_rank),
        "language": lang,
    }


def build_ir_evaluator_payload(
    collection_rows: list[dict],
    query_rows: list[dict],
    language: str = DEFAULT_LANGUAGE,
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """Build corpus, queries, and relevance maps for dev evaluation."""
    lang = normalize_language(language)
    corpus = {str(row["pubkey"]): article_to_text(row) for row in collection_rows}
    queries = {}
    relevant_docs = {}
    for idx, row in enumerate(query_rows):
        pubkey = str(row.get("pubkey") or "")
        if not pubkey:
            continue
        query_id = f"{lang}_dev_{idx}"
        queries[query_id] = str(row["text"]).strip()
        relevant_docs[query_id] = {pubkey}
    return corpus, queries, relevant_docs


def _load_collection_and_split(language: str, split: str):
    from datasets import load_dataset

    lang = normalize_language(language)
    collection_rows = list(
        load_dataset(CHECKTHAT_DATASET, "collection", split="collection")
    )
    query_rows = list(load_dataset(CHECKTHAT_DATASET, lang, split=split))
    return collection_rows, query_rows


def texts_fingerprint(texts: list[str]) -> str:
    """Compute a deterministic fingerprint for cache validation."""
    import hashlib

    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8", errors="ignore")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def load_completed_query_keys(path: str) -> set[str]:
    """Read already-mined query keys from a JSONL triplet file."""
    if not os.path.exists(path):
        return set()

    completed = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            query_key = row.get("query_key") or row.get("query_pubkey")
            if query_key is not None:
                completed.add(str(query_key))
    return completed


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Return the newest Hugging Face Trainer checkpoint under a directory."""
    if not os.path.isdir(checkpoint_dir):
        return None

    candidates = []
    for name in os.listdir(checkpoint_dir):
        if not name.startswith("checkpoint-"):
            continue
        step_text = name.rsplit("-", 1)[-1]
        if not step_text.isdigit():
            continue
        path = os.path.join(checkpoint_dir, name)
        trainer_state_path = os.path.join(path, "trainer_state.json")
        if os.path.isdir(path) and os.path.isfile(trainer_state_path):
            candidates.append((int(step_text), path))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def resolve_resume_checkpoint(
    checkpoint_dir: str,
    resume_from_checkpoint: str | None,
) -> str | None:
    """Resolve an explicit, disabled, or auto checkpoint resume setting."""
    if resume_from_checkpoint is None:
        return None

    normalized = resume_from_checkpoint.strip()
    if normalized.lower() in {"", "none", "false", "0", "off"}:
        return None
    if normalized.lower() == "auto":
        return find_latest_checkpoint(checkpoint_dir)
    return normalized


def mine_harrier_hard_negatives(
    *,
    language: str = DEFAULT_LANGUAGE,
    output_path: str | None = None,
    document_embeddings_path: str | None = None,
    data_dir: str = DATA_MOUNT,
    model_name: str = HARRIER_MODEL_NAME,
    query_prompt: str | None = None,
    negatives_per_query: int = 4,
    search_top_k: int = 100,
    query_batch_size: int = 64,
    document_batch_size: int = 8,
    resume_mining: bool = True,
    force_recompute_document_embeddings: bool = False,
    commit_every_batches: int = 5,
    device: str = "cuda",
    on_cache_update: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Mine hard negatives from current Harrier retrieval false positives."""
    import torch
    from sentence_transformers import SentenceTransformer
    from tqdm.auto import tqdm

    lang = normalize_language(language)
    paths = default_language_paths(data_dir, lang)
    output_path = output_path or paths["triplets_path"]
    document_embeddings_path = (
        document_embeddings_path or paths["document_embeddings_path"]
    )
    query_prompt = query_prompt or query_prompt_for_language(lang)

    if negatives_per_query < 1:
        raise ValueError("negatives_per_query must be at least 1.")
    if search_top_k < negatives_per_query + 1:
        raise ValueError("search_top_k should exceed negatives_per_query.")
    if commit_every_batches < 1:
        raise ValueError("commit_every_batches must be at least 1.")

    print(f"Loading CheckThat collection and {language_name(lang)} train split...")
    collection_rows, train_rows = _load_collection_and_split(lang, "train")
    document_texts = [article_to_text(row) for row in collection_rows]
    pubkeys = [str(row["pubkey"]) for row in collection_rows]
    pubkey_to_text = dict(zip(pubkeys, document_texts, strict=True))
    document_fingerprint = texts_fingerprint(document_texts)

    completed_query_keys = (
        load_completed_query_keys(output_path) if resume_mining else set()
    )
    if completed_query_keys:
        print(f"Resume mining enabled: skipping {len(completed_query_keys)} queries.")

    print(f"Loading Harrier model: {model_name}")
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"dtype": "auto"},
        token=os.environ.get("HF_TOKEN"),
    )

    document_embeddings = None
    if (
        document_embeddings_path
        and os.path.exists(document_embeddings_path)
        and not force_recompute_document_embeddings
    ):
        print(f"Loading cached document embeddings from {document_embeddings_path}")
        cache = torch.load(document_embeddings_path, map_location=device)
        if (
            cache.get("model_name") == model_name
            and cache.get("fingerprint") == document_fingerprint
        ):
            document_embeddings = cache["embeddings"]
        else:
            print("Cached document embeddings do not match current collection/model.")

    if document_embeddings is None:
        print(f"Encoding {len(document_texts)} collection documents...")
        document_embeddings = model.encode(
            document_texts,
            batch_size=document_batch_size,
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        if document_embeddings_path:
            _ensure_parent_dir(document_embeddings_path)
            torch.save(
                {
                    "model_name": model_name,
                    "fingerprint": document_fingerprint,
                    "embeddings": document_embeddings,
                },
                document_embeddings_path,
            )
            if on_cache_update is not None:
                on_cache_update()
            print(f"Saved document embeddings cache to {document_embeddings_path}")

    _ensure_parent_dir(output_path)
    top_k = min(search_top_k + 1, len(document_texts))
    written = 0
    skipped_missing_positive = 0
    skipped_no_negative = 0

    print(f"Mining {lang} hard negatives into {output_path}...")
    output_mode = "a" if resume_mining else "w"
    with open(output_path, output_mode, encoding="utf-8") as output:
        processed_batches_since_commit = 0
        for start in tqdm(range(0, len(train_rows), query_batch_size)):
            batch_items = [
                (idx, row)
                for idx, row in enumerate(
                    train_rows[start : start + query_batch_size], start=start
                )
                if f"{lang}_train_{idx}" not in completed_query_keys
            ]
            batch_rows = [row for _idx, row in batch_items]
            if not batch_rows:
                continue

            query_texts = [str(row["text"]).strip() for row in batch_rows]
            query_embeddings = model.encode(
                query_texts,
                batch_size=query_batch_size,
                convert_to_tensor=True,
                device=device,
                normalize_embeddings=True,
                prompt=query_prompt,
                show_progress_bar=False,
            )
            scores = query_embeddings @ document_embeddings.T
            top_indices = torch.topk(scores, k=top_k, dim=1).indices.cpu().tolist()

            for (query_idx, row), candidate_indices in zip(
                batch_items, top_indices, strict=True
            ):
                query_key = f"{lang}_train_{query_idx}"
                true_pubkey = str(row["pubkey"])
                positive_text = pubkey_to_text.get(true_pubkey)
                if positive_text is None:
                    skipped_missing_positive += 1
                    completed_query_keys.add(query_key)
                    continue

                selected = 0
                for rank, doc_idx in enumerate(candidate_indices, start=1):
                    negative_pubkey = pubkeys[doc_idx]
                    if negative_pubkey == true_pubkey:
                        continue

                    record = make_triplet_record(
                        query_row=row,
                        positive_text=positive_text,
                        negative_text=document_texts[doc_idx],
                        negative_pubkey=negative_pubkey,
                        negative_rank=rank,
                        language=lang,
                        query_key=query_key,
                    )
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    selected += 1
                    if selected >= negatives_per_query:
                        break

                if selected == 0:
                    skipped_no_negative += 1
                completed_query_keys.add(query_key)

            output.flush()
            processed_batches_since_commit += 1
            if (
                on_cache_update is not None
                and processed_batches_since_commit >= commit_every_batches
            ):
                on_cache_update()
                processed_batches_since_commit = 0

    if on_cache_update is not None:
        on_cache_update()
    print(
        "Hard-negative mining complete: "
        f"{written} triplets, {skipped_missing_positive} missing positives, "
        f"{skipped_no_negative} queries without negatives."
    )
    return {
        "language": lang,
        "output_path": output_path,
        "triplets": written,
        "missing_positives": skipped_missing_positive,
        "queries_without_negatives": skipped_no_negative,
    }


def build_training_kwargs(
    *,
    language: str = DEFAULT_LANGUAGE,
    data_dir: str = DATA_MOUNT,
    triplets_path: str | None = None,
    checkpoint_dir: str | None = None,
    output_dir: str | None = None,
    model_name: str = HARRIER_MODEL_NAME,
    query_prompt: str | None = None,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 5e-6,
    warmup_ratio: float = 0.05,
    triplet_margin: float = 0.1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: str = "all-linear",
    max_train_examples: int | None = None,
    save_steps: int = 100,
    eval_steps: int = 1000,
    save_total_limit: int = 12,
    resume_from_checkpoint: str | None = "auto",
    dataloader_num_workers: int = 4,
) -> dict[str, object]:
    """Collect serializable training arguments for local or distributed launch."""
    lang = normalize_language(language)
    paths = default_language_paths(data_dir, lang)
    return {
        "language": lang,
        "triplets_path": triplets_path or paths["triplets_path"],
        "checkpoint_dir": checkpoint_dir or paths["checkpoint_dir"],
        "output_dir": output_dir or paths["output_dir"],
        "model_name": model_name,
        "query_prompt": query_prompt or query_prompt_for_language(lang),
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "triplet_margin": triplet_margin,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "max_train_examples": max_train_examples,
        "save_steps": save_steps,
        "eval_steps": eval_steps,
        "save_total_limit": save_total_limit,
        "resume_from_checkpoint": resume_from_checkpoint,
        "dataloader_num_workers": dataloader_num_workers,
    }


def _distributed_barrier() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def train_harrier_lora(
    *,
    language: str = DEFAULT_LANGUAGE,
    triplets_path: str,
    checkpoint_dir: str,
    output_dir: str,
    model_name: str = HARRIER_MODEL_NAME,
    query_prompt: str | None = None,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 5e-6,
    warmup_ratio: float = 0.05,
    triplet_margin: float = 0.1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: str = "all-linear",
    max_train_examples: int | None = None,
    save_steps: int = 100,
    eval_steps: int = 1000,
    save_total_limit: int = 12,
    resume_from_checkpoint: str | None = "auto",
    dataloader_num_workers: int = 4,
    on_checkpoint_save: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Train a Harrier LoRA adapter for one language."""
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, TaskType
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.losses import TripletDistanceMetric, TripletLoss
    from transformers import TrainerCallback

    lang = normalize_language(language)
    query_prompt = query_prompt or query_prompt_for_language(lang)

    if not os.path.exists(triplets_path):
        raise FileNotFoundError(
            f"Triplets not found at {triplets_path}. Run mining first."
        )
    if save_steps < 1:
        raise ValueError("save_steps must be at least 1.")
    if eval_steps < 1:
        raise ValueError("eval_steps must be at least 1.")
    if save_total_limit < 1:
        raise ValueError("save_total_limit must be at least 1.")

    print(f"Loading Harrier base model for {language_name(lang)}: {model_name}")
    model = SentenceTransformer(
        model_name,
        device="cuda",
        model_kwargs={"dtype": "auto"},
        token=os.environ.get("HF_TOKEN"),
    )

    print("Attaching LoRA adapter...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=parse_target_modules(target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model.add_adapter(lora_config)
    model.max_seq_length = 2048

    print(f"Loading triplets from {triplets_path}")
    train_dataset = load_dataset("json", data_files=triplets_path, split="train")
    keep_columns = {"anchor", "positive", "negative"}
    drop_columns = [
        col for col in train_dataset.column_names if col not in keep_columns
    ]
    if drop_columns:
        train_dataset = train_dataset.remove_columns(drop_columns)
    if max_train_examples is not None:
        train_dataset = train_dataset.select(
            range(min(max_train_examples, len(train_dataset)))
        )

    def apply_query_prompt(row):
        row["anchor"] = f"{query_prompt}{row['anchor']}"
        return row

    train_dataset = train_dataset.map(apply_query_prompt)

    print(f"Preparing {language_name(lang)} dev evaluator...")
    collection_rows, dev_rows = _load_collection_and_split(lang, "dev")
    corpus, queries, relevant_docs = build_ir_evaluator_payload(
        collection_rows,
        dev_rows,
        language=lang,
    )
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=f"{lang}_dev",
        batch_size=4,
        show_progress_bar=True,
        query_prompt=query_prompt,
        main_score_function="cosine",
    )
    load_best_model_at_end = save_steps == eval_steps
    if not load_best_model_at_end:
        print(
            "save_steps and eval_steps differ; keeping latest checkpoint/final "
            "adapter instead of reloading best dev checkpoint."
        )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=f"eval_{lang}_dev_cosine_ndcg@10",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=2 if dataloader_num_workers > 0 else None,
        dataloader_persistent_workers=dataloader_num_workers > 0,
        run_name=f"harrier-27b-{lang}-lora",
    )

    class DurableCheckpointCallback(TrainerCallback):
        """Run a provider-specific durability hook after checkpoint saves."""

        def on_save(self, args, state, control, **kwargs):
            if on_checkpoint_save is not None:
                on_checkpoint_save()
                print(
                    f"Completed checkpoint durability hook at step {state.global_step}."
                )
            return control

    loss = TripletLoss(
        model=model,
        distance_metric=TripletDistanceMetric.COSINE,
        triplet_margin=triplet_margin,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        evaluator=evaluator,
        loss=loss,
        callbacks=[DurableCheckpointCallback()],
    )

    resume_checkpoint = resolve_resume_checkpoint(
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
    else:
        print("Starting training from base model; no checkpoint resume selected/found.")

    print(f"Starting {language_name(lang)} Harrier LoRA training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        print(f"Reloading best adapter checkpoint: {best_ckpt}")
        model[0].auto_model = PeftModel.from_pretrained(
            model[0].auto_model,
            best_ckpt,
        )
    else:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt:
            print(
                f"No best checkpoint selected; latest durable checkpoint: {latest_ckpt}"
            )

    if trainer.is_world_process_zero():
        print(f"Saving final adapter to {output_dir}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        if on_checkpoint_save is not None:
            on_checkpoint_save()
    _distributed_barrier()

    print(f"Harrier {language_name(lang)} fine-tuning complete.")
    return {
        "language": lang,
        "output_dir": output_dir,
        "best_checkpoint": best_ckpt,
    }


def run_distributed_worker(args_path: str) -> dict[str, object]:
    """Run one Accelerate-launched training worker from serialized args."""
    with open(args_path, encoding="utf-8") as handle:
        kwargs = json.load(handle)
    return train_harrier_lora(**kwargs)


def launch_distributed_training(
    *,
    training_kwargs: dict[str, object],
    args_path: str,
    num_processes: int = 4,
    mixed_precision: str = "bf16",
) -> dict[str, object]:
    """Launch single-node multi-GPU training through Accelerate."""
    if num_processes < 2:
        return train_harrier_lora(**training_kwargs)

    _ensure_parent_dir(args_path)
    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(training_kwargs, handle, ensure_ascii=False, indent=2)

    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(num_processes),
        "--num_machines",
        "1",
        "--mixed_precision",
        mixed_precision,
        "--multi_gpu",
        "-m",
        "clef_training.train_harrier_language",
        "--distributed-worker",
        args_path,
    ]
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    print("Launching distributed training:")
    print(" ".join(command))
    subprocess.run(command, check=True, env=env)
    return {"args_path": args_path, "processes": num_processes}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mine", "train", "all"), default="all")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--data-dir", default=DATA_MOUNT)
    parser.add_argument("--distributed-worker")
    parser.add_argument("--model-name", default=HARRIER_MODEL_NAME)
    parser.add_argument("--query-prompt")

    parser.add_argument("--triplets-path")
    parser.add_argument("--document-embeddings-path")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--distributed-args-path")

    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--search-top-k", type=int, default=100)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--document-batch-size", type=int, default=8)
    parser.add_argument("--resume-mining", dest="resume_mining", action="store_true")
    parser.add_argument(
        "--no-resume-mining", dest="resume_mining", action="store_false"
    )
    parser.set_defaults(resume_mining=True)
    parser.add_argument("--force-recompute-document-embeddings", action="store_true")
    parser.add_argument("--commit-every-batches", type=int, default=5)

    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--triplet-margin", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=12)
    parser.add_argument("--resume-from-checkpoint", default="auto")
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.distributed_worker:
        run_distributed_worker(args.distributed_worker)
        return

    lang = normalize_language(args.language)
    paths = default_language_paths(args.data_dir, lang)
    triplets_path = args.triplets_path or paths["triplets_path"]
    document_embeddings_path = (
        args.document_embeddings_path or paths["document_embeddings_path"]
    )
    checkpoint_dir = args.checkpoint_dir or paths["checkpoint_dir"]
    output_dir = args.output_dir or paths["output_dir"]
    distributed_args_path = args.distributed_args_path or paths["distributed_args_path"]

    if args.mode in {"mine", "all"}:
        mine_harrier_hard_negatives(
            language=lang,
            output_path=triplets_path,
            document_embeddings_path=document_embeddings_path,
            model_name=args.model_name,
            query_prompt=args.query_prompt,
            negatives_per_query=args.negatives_per_query,
            search_top_k=args.search_top_k,
            query_batch_size=args.query_batch_size,
            document_batch_size=args.document_batch_size,
            resume_mining=args.resume_mining,
            force_recompute_document_embeddings=(
                args.force_recompute_document_embeddings
            ),
            commit_every_batches=args.commit_every_batches,
            device=args.device,
        )

    if args.mode in {"train", "all"}:
        training_kwargs = build_training_kwargs(
            language=lang,
            data_dir=args.data_dir,
            triplets_path=triplets_path,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            model_name=args.model_name,
            query_prompt=args.query_prompt,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            triplet_margin=args.triplet_margin,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            max_train_examples=args.max_train_examples,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=args.save_total_limit,
            resume_from_checkpoint=args.resume_from_checkpoint,
            dataloader_num_workers=args.dataloader_num_workers,
        )
        launch_distributed_training(
            training_kwargs=training_kwargs,
            args_path=distributed_args_path,
            num_processes=args.num_processes,
            mixed_precision=args.mixed_precision,
        )


if __name__ == "__main__":
    main()
