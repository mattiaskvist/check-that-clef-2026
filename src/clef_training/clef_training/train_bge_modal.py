import modal

# 1. Define the container image and install required libraries
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "sentence-transformers", "peft>=0.10", "datasets>=2.16", "accelerate>=0.28"
    )
)

app = modal.App("checkthat-bge-m3-training")

# 2. Create a persistent Volume to save your checkpoints and final model
volume = modal.Volume.from_name("clef-vol", create_if_missing=True)


# 3. Define the training function and request GPU resources
@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60 * 12,  # Set a 12-hour timeout limit
    volumes={"/data": volume},  # Mount the persistent volume to the /data directory
    secrets=[modal.Secret.from_name("hf-token")],
)
def train_model():
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, PeftModel
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    print("Loading base model...")
    model = SentenceTransformer("BAAI/bge-m3")

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "key", "value"],
        layers_to_transform=[20, 21, 22, 23],  # Targets only the last 4 layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model.add_adapter(lora_config)
    model.max_seq_length = 1024

    # Load dataset from the persistent volume
    print("Loading dataset...")
    train_dataset = load_dataset(
        "json", data_files="/data/hard_negative_triplets.json", split="train"
    )

    # --- 2. Load and Prepare Evaluation Data ---
    print("Loading evaluation datasets...")
    # Load the 10,000 document collection
    collection_dataset = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        "collection",
        split="collection",
    )

    # Load the official dev splits
    en_dev = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        "en",
        split="dev",
    )
    de_dev = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        "de",
        split="dev",
    )
    fr_dev = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        "fr",
        split="dev",
    )

    corpus = {}
    queries = {}
    relevant_docs = {}

    # Build the corpus dictionary mapping pubkey -> Title + Abstract
    for doc in collection_dataset:
        corpus[doc["pubkey"]] = f"{doc['title']}\n{doc['abstract']}"

    # Helper function to populate the queries and relevant_docs dictionaries
    def process_dev_queries(dev_data, lang_prefix):
        for idx, row in enumerate(dev_data):
            # Create a unique query ID since the raw data might not have one
            query_id = f"{lang_prefix}_dev_{idx}"
            queries[query_id] = row["text"]
            # Evaluator requires a set() of relevant document IDs
            relevant_docs[query_id] = {row["pubkey"]}

    process_dev_queries(en_dev, "en")
    process_dev_queries(de_dev, "de")
    process_dev_queries(fr_dev, "fr")

    # --- 3. Initialize the Evaluator ---
    print("Initializing InformationRetrievalEvaluator...")
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="multilingual_dev",
        show_progress_bar=True,
    )

    loss = MultipleNegativesRankingLoss(model)

    # Configure the trainer to save checkpoints to the mounted /data/ directory
    training_args = SentenceTransformerTrainingArguments(
        output_dir="/data/bge-m3-checkthat-checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=256,
        auto_find_batch_size=True,
        learning_rate=2e-4,
        warmup_steps=0.1,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        gradient_checkpointing=True,  # Enable gradient checkpointing to reduce memory usage
        metric_for_best_model="eval_multilingual_dev_cosine_ndcg@10",
        load_best_model_at_end=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        evaluator=evaluator,
        loss=loss,
    )

    print("Starting training...")
    trainer.train()

    # load_best_model_at_end should have already loaded the best weights,
    # but we need to ensure the PEFT adapters are correctly mapped
    # We will still get warnings about missing keys,
    # but this is expected due to the way PEFT modifies the model architecture

    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        print(
            f"\nHealing model prefix bug. Loading true best weights from: {best_ckpt}"
        )
        # use the PEFT library to correctly map the weights
        model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, best_ckpt)
    else:
        print("\nWarning: Could not identify best checkpoint.")

    print("Saving the best model to /data/bge-m3-finetuned...")
    model.save_pretrained("/data/bge-m3-finetuned")

    volume.commit()
    print("Training complete!")


# 4. Entry point to trigger the remote run
@app.local_entrypoint()
def main():
    train_model.remote()
