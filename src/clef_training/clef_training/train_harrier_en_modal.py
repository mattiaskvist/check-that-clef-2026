"""English-only Harrier hard-negative mining and LoRA fine-tuning workflow."""

from __future__ import annotations

import json

import modal


CHECKTHAT_DATASET = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"
HARRIER_MODEL_NAME = "microsoft/harrier-oss-v1-27b"
EN_QUERY_PROMPT = (
    "Instruct: Retrieve the implicitly referenced medical-scientific publication "
    "for this English claim. Prioritize diseases, interventions, populations, "
    "measurements, acronyms, study names, and exact biomedical terms.\nQuery: "
)

DATA_MOUNT = "/data"
VOLUME_NAME = "clef-vol"
DEFAULT_TRIPLETS_PATH = f"{DATA_MOUNT}/harrier_en_hard_negatives.jsonl"
DEFAULT_DOCUMENT_EMBEDDINGS_PATH = f"{DATA_MOUNT}/harrier_en_document_embeddings.pt"
DEFAULT_CHECKPOINT_DIR = f"{DATA_MOUNT}/harrier-27b-en-checkpoints"
DEFAULT_OUTPUT_DIR = f"{DATA_MOUNT}/harrier-27b-en-lora"
DEFAULT_DISTRIBUTED_ARGS_PATH = f"{DATA_MOUNT}/harrier_en_train_args.json"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "accelerate>=0.28",
        "datasets>=4.8.4",
        "peft>=0.18.1",
        "sentence-transformers==5.3.0",
        "tqdm",
        # Harrier's config was authored against Transformers 4.57.6. Newer
        # Gemma3 loaders may try to construct an image processor for this
        # text-only embedding checkpoint and fail due to missing
        # preprocessor_config.json.
        "transformers==4.57.6",
    )
)

app = modal.App("checkthat-harrier-en-training")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


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
    query_key: str | None = None,
) -> dict[str, object]:
    """Build one JSONL training record with metadata for auditing."""
    resolved_query_key = query_key or str(query_row.get("index") or query_row["pubkey"])
    return {
        "query_key": resolved_query_key,
        "anchor": str(query_row["text"]).strip(),
        "positive": positive_text,
        "negative": negative_text,
        "query_pubkey": str(query_row["pubkey"]),
        "negative_pubkey": str(negative_pubkey),
        "negative_rank": int(negative_rank),
        "language": "en",
    }


def build_ir_evaluator_payload(
    collection_rows: list[dict], query_rows: list[dict]
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """Build corpus, queries, and relevance maps for English dev evaluation."""
    corpus = {str(row["pubkey"]): article_to_text(row) for row in collection_rows}
    queries = {}
    relevant_docs = {}
    for idx, row in enumerate(query_rows):
        pubkey = str(row.get("pubkey") or "")
        if not pubkey:
            continue
        query_id = f"en_dev_{idx}"
        queries[query_id] = str(row["text"]).strip()
        relevant_docs[query_id] = {pubkey}
    return corpus, queries, relevant_docs


def _load_collection_and_split(split: str):
    from datasets import load_dataset

    collection_rows = list(
        load_dataset(CHECKTHAT_DATASET, "collection", split="collection")
    )
    query_rows = list(load_dataset(CHECKTHAT_DATASET, "en", split=split))
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
    import os

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
    import os

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
    checkpoint_dir: str, resume_from_checkpoint: str | None
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


def build_training_kwargs(
    *,
    triplets_path: str = DEFAULT_TRIPLETS_PATH,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_name: str = HARRIER_MODEL_NAME,
    query_prompt: str = EN_QUERY_PROMPT,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
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
    return {
        "triplets_path": triplets_path,
        "checkpoint_dir": checkpoint_dir,
        "output_dir": output_dir,
        "model_name": model_name,
        "query_prompt": query_prompt,
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


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 10,
    volumes={DATA_MOUNT: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
def mine_harrier_en_hard_negatives(
    output_path: str = DEFAULT_TRIPLETS_PATH,
    document_embeddings_path: str = DEFAULT_DOCUMENT_EMBEDDINGS_PATH,
    negatives_per_query: int = 4,
    search_top_k: int = 100,
    query_batch_size: int = 64,
    document_batch_size: int = 8,
    resume_mining: bool = True,
    force_recompute_document_embeddings: bool = False,
    commit_every_batches: int = 5,
):
    """Mine English hard negatives from current Harrier 27B false positives."""
    import os

    import torch
    from sentence_transformers import SentenceTransformer
    from tqdm.auto import tqdm

    if negatives_per_query < 1:
        raise ValueError("negatives_per_query must be at least 1.")
    if search_top_k < negatives_per_query + 1:
        raise ValueError("search_top_k should exceed negatives_per_query.")
    if commit_every_batches < 1:
        raise ValueError("commit_every_batches must be at least 1.")

    volume.reload()
    print("Loading CheckThat collection and English train split...")
    collection_rows, train_rows = _load_collection_and_split("train")
    document_texts = [article_to_text(row) for row in collection_rows]
    pubkeys = [str(row["pubkey"]) for row in collection_rows]
    pubkey_to_text = dict(zip(pubkeys, document_texts, strict=True))
    document_fingerprint = texts_fingerprint(document_texts)

    completed_query_keys = (
        load_completed_query_keys(output_path) if resume_mining else set()
    )
    if completed_query_keys:
        print(f"Resume mining enabled: skipping {len(completed_query_keys)} queries.")

    print(f"Loading Harrier model: {HARRIER_MODEL_NAME}")
    model = SentenceTransformer(
        HARRIER_MODEL_NAME,
        device="cuda",
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
        cache = torch.load(document_embeddings_path, map_location="cuda")
        if (
            cache.get("model_name") == HARRIER_MODEL_NAME
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
            device="cuda",
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        if document_embeddings_path:
            os.makedirs(os.path.dirname(document_embeddings_path), exist_ok=True)
            torch.save(
                {
                    "model_name": HARRIER_MODEL_NAME,
                    "fingerprint": document_fingerprint,
                    "embeddings": document_embeddings,
                },
                document_embeddings_path,
            )
            volume.commit()
            print(f"Saved document embeddings cache to {document_embeddings_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    top_k = min(search_top_k + 1, len(document_texts))
    written = 0
    skipped_missing_positive = 0
    skipped_no_negative = 0

    print(f"Mining hard negatives into {output_path}...")
    output_mode = "a" if resume_mining else "w"
    with open(output_path, output_mode, encoding="utf-8") as output:
        processed_batches_since_commit = 0
        for start in tqdm(range(0, len(train_rows), query_batch_size)):
            batch_items = [
                (idx, row)
                for idx, row in enumerate(
                    train_rows[start : start + query_batch_size], start=start
                )
                if f"en_train_{idx}" not in completed_query_keys
            ]
            batch_rows = [row for _idx, row in batch_items]
            if not batch_rows:
                continue

            query_texts = [str(row["text"]).strip() for row in batch_rows]
            query_embeddings = model.encode(
                query_texts,
                batch_size=query_batch_size,
                convert_to_tensor=True,
                device="cuda",
                normalize_embeddings=True,
                prompt=EN_QUERY_PROMPT,
                show_progress_bar=False,
            )
            scores = query_embeddings @ document_embeddings.T
            top_indices = torch.topk(scores, k=top_k, dim=1).indices.cpu().tolist()

            for (query_idx, row), candidate_indices in zip(
                batch_items, top_indices, strict=True
            ):
                query_key = f"en_train_{query_idx}"
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
            if processed_batches_since_commit >= commit_every_batches:
                volume.commit()
                processed_batches_since_commit = 0

    volume.commit()
    print(
        "Hard-negative mining complete: "
        f"{written} triplets, {skipped_missing_positive} missing positives, "
        f"{skipped_no_negative} queries without negatives."
    )
    return {
        "output_path": output_path,
        "triplets": written,
        "missing_positives": skipped_missing_positive,
        "queries_without_negatives": skipped_no_negative,
    }


def _train_harrier_en_lora_impl(
    triplets_path: str = DEFAULT_TRIPLETS_PATH,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_name: str = HARRIER_MODEL_NAME,
    query_prompt: str = EN_QUERY_PROMPT,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
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
):
    """Train a Harrier LoRA adapter on English hard-negative triplets."""
    import os

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

    volume.reload()
    print(f"Loading Harrier base model: {model_name}")
    model_kwargs = {"dtype": "auto"}

    model = SentenceTransformer(
        model_name,
        device="cuda",
        model_kwargs=model_kwargs,
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

    print("Preparing English dev evaluator...")
    collection_rows, dev_rows = _load_collection_and_split("dev")
    corpus, queries, relevant_docs = build_ir_evaluator_payload(
        collection_rows, dev_rows
    )
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="en_dev",
        batch_size=4,
        show_progress_bar=True,
        query_prompt=query_prompt,
        main_score_function="cosine",
    )
    load_best_model_at_end = save_steps == eval_steps
    if not load_best_model_at_end:
        print(
            "save_steps and eval_steps differ; keeping latest checkpoint/final adapter "
            "instead of reloading best dev checkpoint."
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
        metric_for_best_model="eval_en_dev_cosine_ndcg@10",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=2 if dataloader_num_workers > 0 else None,
        dataloader_persistent_workers=dataloader_num_workers > 0,
    )

    class ModalVolumeCommitCallback(TrainerCallback):
        """Commit Modal volume writes after every Trainer checkpoint save."""

        def on_save(self, args, state, control, **kwargs):
            volume.commit()
            print(f"Committed Modal volume after checkpoint step {state.global_step}.")
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
        callbacks=[ModalVolumeCommitCallback()],
    )

    resume_checkpoint = resolve_resume_checkpoint(
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
    else:
        print("Starting training from base model; no checkpoint resume selected/found.")

    print("Starting English Harrier LoRA training...")
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

    print(f"Saving final adapter to {output_dir}")
    model.save_pretrained(output_dir)
    volume.commit()
    print("Harrier English fine-tuning complete.")
    return {"output_dir": output_dir, "best_checkpoint": best_ckpt}


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 24,
    volumes={DATA_MOUNT: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
def train_harrier_en_lora(**kwargs):
    """Train a Harrier LoRA adapter on one B200."""
    return _train_harrier_en_lora_impl(**kwargs)


@app.function(
    image=image,
    gpu="B200:2",
    timeout=60 * 60 * 24,
    volumes={DATA_MOUNT: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
def train_harrier_en_lora_multi_gpu(**kwargs):
    """Launch distributed two-GPU Harrier LoRA training with Accelerate."""
    import os
    import subprocess
    import sys

    volume.reload()
    os.makedirs(os.path.dirname(DEFAULT_DISTRIBUTED_ARGS_PATH), exist_ok=True)
    with open(DEFAULT_DISTRIBUTED_ARGS_PATH, "w", encoding="utf-8") as handle:
        json.dump(kwargs, handle, ensure_ascii=False)
    volume.commit()

    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "2",
        "--multi_gpu",
        "-m",
        "clef_training.train_harrier_en_modal",
        "--distributed-worker",
        DEFAULT_DISTRIBUTED_ARGS_PATH,
    ]
    print("Launching distributed training:")
    print(" ".join(command))
    subprocess.run(command, check=True)
    volume.commit()
    return {"args_path": DEFAULT_DISTRIBUTED_ARGS_PATH, "processes": 2}


def run_distributed_worker(args_path: str):
    """Run one Accelerate-launched training worker from serialized args."""
    volume.reload()
    with open(args_path, encoding="utf-8") as handle:
        kwargs = json.load(handle)
    return _train_harrier_en_lora_impl(**kwargs)


@app.local_entrypoint()
def main(
    mode: str = "all",
    distributed_worker: str | None = None,
    negatives_per_query: int = 4,
    search_top_k: int = 100,
    triplets_path: str = DEFAULT_TRIPLETS_PATH,
    document_embeddings_path: str = DEFAULT_DOCUMENT_EMBEDDINGS_PATH,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_train_examples: int | None = None,
    save_steps: int = 100,
    eval_steps: int = 1000,
    save_total_limit: int = 12,
    resume_from_checkpoint: str | None = "auto",
    multi_gpu: bool = False,
    resume_mining: bool = True,
):
    """Run mining, training, or both on Modal."""
    if distributed_worker:
        run_distributed_worker(distributed_worker)
        return

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"mine", "train", "all"}:
        raise ValueError("mode must be one of: mine, train, all")

    if normalized_mode in {"mine", "all"}:
        mine_harrier_en_hard_negatives.remote(
            output_path=triplets_path,
            document_embeddings_path=document_embeddings_path,
            negatives_per_query=negatives_per_query,
            search_top_k=search_top_k,
            resume_mining=resume_mining,
        )

    if normalized_mode in {"train", "all"}:
        train_fn = (
            train_harrier_en_lora_multi_gpu if multi_gpu else train_harrier_en_lora
        )
        train_fn.remote(
            triplets_path=triplets_path,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            max_train_examples=max_train_examples,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_total_limit=save_total_limit,
            resume_from_checkpoint=resume_from_checkpoint,
        )
