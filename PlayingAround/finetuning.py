import json
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template

# ==========================================
# 1. Configuration & Model Loading
# ==========================================
max_seq_length = 1024 
model_name = "unsloth/Llama-3.2-1B-Instruct" 

print("Loading model in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True, # Critical for 8GB VRAM (RTX 4060)
    dtype=None,
)

# Apply Llama 3 chat template format
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Add LoRA adapters (this is what we actually train)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth", 
)

# ==========================================
# 2. Data Preparation
# ==========================================
print("Loading datasets and mapping ground truth...")

collection_data = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "collection")["collection"]

pubkey_to_truth = {}
for pk, t, a in zip(collection_data["pubkey"], collection_data["title"], collection_data["authors"]):
    pubkey_to_truth[pk] = {"title": t if t else "", "authors": a if a else ""}

train_tweets = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "en")["train"]

def formatting_prompts_func(examples):
    texts = []
    for tweet, pubkey in zip(examples["text"], examples["pubkey"]):
        truth = pubkey_to_truth.get(pubkey, {"title": "", "authors": ""})
        target_json = json.dumps(truth)
        
        # System prompt explicitly removes keyterms and asks for JSON
        messages = [
            {"role": "system", "content": "Extract the referenced research paper from the tweet. Return ONLY a valid JSON object with 'title' and 'authors'."},
            {"role": "user", "content": f"Tweet: {tweet}"},
            {"role": "assistant", "content": target_json}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

# Apply formatting to dataset
formatted_dataset = train_tweets.map(formatting_prompts_func, batched=True)

# ==========================================
# 3. Training
# ==========================================
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=150, # Adjust higher (e.g., 300-500) for a full run
        learning_rate=2e-4,
        fp16=not FastLanguageModel.is_bfloat16_supported(),
        bf16=FastLanguageModel.is_bfloat16_supported(),
        logging_steps=10,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=3407,
    ),
)

trainer.train()

# ==========================================
# 4. Export to Ollama (GGUF)
# ==========================================
print("Exporting to GGUF format for Ollama...")
# This will save a quantized GGUF file in the 'distilled_model' folder
model.save_pretrained_gguf("distilled_model", tokenizer, quantization_method="q4_k_m")
print("✅ Export complete!")