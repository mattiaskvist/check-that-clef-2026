from sentence_transformers import SentenceTransformer, util
from peft import PeftModel

# 1. Load the original base model
print("Loading base model...")
model = SentenceTransformer("BAAI/bge-m3")
hf_id = "mattiaskvist/bge-m3-checkthat-finetuned"  # NOTE: this is currently private, we need to clean up some stuff before making it public.

# 2. Use PEFT directly to inject the weights into the underlying transformer
print("Injecting LoRA adapters...")
model[0].auto_model = PeftModel.from_pretrained(
    model[0].auto_model,
    hf_id,
)

print("Model is ready for retrieval!")

print("\n--- Running Verification Test ---")

# 1. Define a mock tweet
query = "Climate change significantly impacts marine biodiversity."

# 2. Define a relevant and an irrelevant document (Title + Abstract format)
documents = [
    "Title: Ocean Warming and Species Migration\nAbstract: This study demonstrates how rising global ocean temperatures are forcing marine species to migrate towards the poles, drastically altering coastal biodiversity.",
    "Title: The Fall of the Roman Empire\nAbstract: A historical overview of the political instability and economic decline that led to the collapse of ancient Rome.",
]

# 3. Generate the dense embeddings!
print("Encoding texts...")
query_embedding = model.encode(query)
doc_embeddings = model.encode(documents)

# 4. Calculate Cosine Similarity between the query and both documents
scores = util.cos_sim(query_embedding, doc_embeddings)[0]

# 5. Display the results
print("\nResults:")
for i, score in enumerate(scores):
    print(f"Document {i + 1} Score: {score:.4f}  |  {documents[i][:45]}...")
