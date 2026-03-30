import os
import json
import numpy as np
import ollama
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = 'llama3.2' 
LANG = "en"
NUM_TWEETS_TO_EVALUATE = 15
EMBEDDINGS_CACHE_FILE = "collection_embeddings.npy"

print("==========================================")
print(f"Checking for model: {MODEL_NAME}")
try:
    ollama.show(MODEL_NAME)
    print("✅ Model is available locally.")
except Exception:
    print(f"⏳ Model not found. Pulling '{MODEL_NAME}'...")
    ollama.pull(MODEL_NAME)
    print("✅ Model pulled successfully!")
print("==========================================\n")

print("Loading datasets...")

# 1. Load Collection
collection_data = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "collection")["collection"]

title_to_pubkey = {}
pubkey_to_title = {}
pubkey_to_abstract = {}

for title, pubkey, abstract in zip(collection_data["title"], collection_data["pubkey"], collection_data["abstract"]):
    if title:
        title_to_pubkey[title.lower()] = pubkey
        pubkey_to_title[pubkey] = title
        pubkey_to_abstract[pubkey] = abstract if abstract else ""

collection_titles = list(title_to_pubkey.keys())

# 2. Build or Load Embeddings Cache
if os.path.exists(EMBEDDINGS_CACHE_FILE):
    print("Loading cached collection embeddings...")
    collection_embeddings = np.load(EMBEDDINGS_CACHE_FILE)
else:
    print("Generating embeddings for the entire collection...")
    print("NOTE: This only happens once. It will be cached for future runs.")
    
    embeddings_list = []
    # Wrap in tqdm for a progress bar since this takes time
    for title in tqdm(collection_titles, desc="Embedding Titles"):
        response = ollama.embeddings(model=MODEL_NAME, prompt=title)
        embeddings_list.append(response['embedding'])
        
    collection_embeddings = np.array(embeddings_list)
    np.save(EMBEDDINGS_CACHE_FILE, collection_embeddings)
    print("✅ Embeddings cached successfully!")

# 3. Load Tweets
data = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", LANG) 
dev_split = data["dev"]

# ==========================================
# Core Functions
# ==========================================

json_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "authors": {"type": "string"},
        "keyterms": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["title", "authors", "keyterms"]
}

def get_llm_extraction(tweet_text):
    prompt = (
        "Read the following tweet and extract the referenced research paper. "
        "Return the title, authors, and a list of 3 to 5 specific keyterms or concepts mentioned. "
        "\n\n"
        f"Tweet: {tweet_text}"
    )
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            format=json_schema, 
            options={"temperature": 0.0} 
        )
        result = json.loads(response['message']['content'])
        return result.get('title', ''), result.get('authors', ''), result.get('keyterms', [])
    except Exception as e:
        print(f"Extraction Error: {e}")
        return "", "", []

def count_keyterm_matches(search_terms, abstract):
    if not abstract or not search_terms:
        return 0
    abstract_lower = abstract.lower()
    matches = 0
    for term in search_terms:
        if term in abstract_lower:
            matches += 1
    return matches

def find_best_matches_embedded(extracted_title, search_terms, true_pubkey):
    if not extracted_title:
        return None
        
    # 1. Generate embedding for the extracted title
    query_response = ollama.embeddings(model=MODEL_NAME, prompt=extracted_title.lower())
    query_embedding = np.array(query_response['embedding'])
    
    # 2. Calculate Cosine Similarity across the whole collection
    # Formula: dot(A, B) / (norm(A) * norm(B))
    dot_products = np.dot(collection_embeddings, query_embedding)
    norms = np.linalg.norm(collection_embeddings, axis=1) * np.linalg.norm(query_embedding)
    similarities = dot_products / norms
    
    # 3. Get the indices of the Top 5 highest similarity scores
    top_5_indices = np.argsort(similarities)[-5:][::-1]
    
    candidates = []
    
    # 4. Gather data for the Top 5
    for idx in top_5_indices:
        match_title_lower = collection_titles[idx]
        sim_score = similarities[idx]
        pubkey = title_to_pubkey[match_title_lower]
        abstract = pubkey_to_abstract.get(pubkey, "")
        
        kw_matches = count_keyterm_matches(search_terms, abstract)
        is_correct = (pubkey == true_pubkey)
        display_title = pubkey_to_title[pubkey]
        
        candidates.append({
            "title": display_title,
            "pubkey": pubkey,
            "sim_score": sim_score,
            "kw_matches": kw_matches,
            "is_correct": is_correct
        })
        
    # 5. Re-rank the candidates
    # We sort primarily by kw_matches (descending), and break ties using sim_score (descending)
    candidates.sort(key=lambda x: (x["kw_matches"], x["sim_score"]), reverse=True)
    
    return candidates

def evaluate_system(tweets_dataset, num_samples):
    correct_matches = 0
    total_processed = 0
    limit = min(num_samples, len(tweets_dataset["text"]))
    
    print(f"\nStarting evaluation on {limit} tweets using {MODEL_NAME} embeddings...\n")
    
    for i in range(limit):
        tweet_text = tweets_dataset["text"][i] 
        true_pubkey = tweets_dataset["pubkey"][i]
        
        print(f"[{i+1}/{limit}] Tweet: {tweet_text[:100]}...")
        
        # Step 1: LLM Extraction
        extracted_title, extracted_authors, raw_keyterms = get_llm_extraction(tweet_text)
        
        # Process Keyterms 
        processed_terms = set()
        for term in raw_keyterms:
            for word in term.split():
                clean_word = word.strip('.,!?()"\'#').lower()
                if len(clean_word) > 1: 
                    processed_terms.add(clean_word)
                    
        print(f"   LLM Title Guessed : '{extracted_title}'")
        print(f"   Search Terms      : {list(processed_terms)}")
        
        # Step 2: Get Ranked Candidates
        ranked_candidates = find_best_matches_embedded(extracted_title, processed_terms, true_pubkey)
        
        if ranked_candidates:
            print("\n   Top 5 Candidates (Ranked by System):")
            for rank, candidate in enumerate(ranked_candidates, 1):
                # Add a marker if this is the ground truth correct paper
                marker = "[CORRECT]" if candidate["is_correct"] else "[       ]"
                title_preview = candidate["title"][:60] + ("..." if len(candidate["title"]) > 60 else "")
                
                print(f"   {rank}. {marker} Sim: {candidate['sim_score']:.4f} | KW: {candidate['kw_matches']} | Title: '{title_preview}'")
            
            # The system's final choice is the #1 ranked item
            final_choice = ranked_candidates[0]
            if final_choice["is_correct"]:
                print("\n   Result: ✅ CORRECT PAPER SELECTED")
                correct_matches += 1
            else:
                print("\n   Result: ❌ MISMATCH")
                
        else:
            print("   Result: ❌ MISMATCH (No title extracted)")
            
        print("-" * 50)
        total_processed += 1
        
    accuracy = (correct_matches / total_processed) * 100 if total_processed > 0 else 0
    
    print("="*40)
    print("      FINAL EVALUATION RESULTS")
    print("="*40)
    print(f"Total Processed: {total_processed}")
    print(f"Correct Matches: {correct_matches}")
    print(f"Accuracy:        {accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    # You will need to install numpy and tqdm if you haven't already:
    # pip install numpy tqdm
    evaluate_system(dev_split, NUM_TWEETS_TO_EVALUATE)