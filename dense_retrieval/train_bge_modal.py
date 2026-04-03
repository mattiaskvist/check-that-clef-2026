import modal

# 1. Define the container image and install required libraries
image = (
    modal.Image.debian_slim(python_version="3.11")
   .pip_install(
        "sentence-transformers",
        "peft>=0.10",
        "datasets>=2.16",
        "accelerate>=0.28"
    )
)

app = modal.App("checkthat-bge-m3-training")

# 2. Create a persistent Volume to save your checkpoints and final model
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)

# 3. Define the training function and request GPU resources
@app.function(
    image=image, 
    gpu="A10G", # Request an NVIDIA GPU (can be upgraded to "A100" if needed)
    timeout=86400, # Set a 24-hour timeout limit
    volumes={"/data": volume} # Mount the persistent volume to the /data directory
)
def train_model():
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    print("Loading base model...")
    model = SentenceTransformer("BAAI/bge-m3")

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    model.auto_model = get_peft_model(model.auto_model, lora_config)
    model.auto_model.print_trainable_parameters()

    # Load dataset from the persistent volume
    print("Loading dataset...")
    train_dataset = load_dataset("json", data_files="/data/your_hard_negative_triplets.json", split="train")
    
    loss = MultipleNegativesRankingLoss(model)

    # Configure the trainer to save checkpoints to the mounted /data/ directory
    training_args = SentenceTransformerTrainingArguments(
        output_dir="/data/bge-m3-checkthat-checkpoints", 
        num_train_epochs=3,
        per_device_train_batch_size=8, 
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        fp16=True, 
        save_strategy="epoch",
        logging_steps=50,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    print("Starting training...")
    trainer.train()

    # Save the final fine-tuned adapter weights to the volume
    print("Saving model...")
    model.save_pretrained("/data/bge-m3-finetuned")
    
    # Ensure all changes are fully persisted to the Volume before the container exits
    volume.commit() 
    print("Training complete!")

# 4. Entry point to trigger the remote run
@app.local_entrypoint()
def main():
    train_model.remote()