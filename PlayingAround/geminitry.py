import os
import difflib
from datasets import load_dataset
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = 'gemini-3-flash-preview' 
LANG = "en"
NUM_TWEETS_TO_EVALUATE = 15

load_dotenv()

client = genai.Client()

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

# 2. Load Tweets
data = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", LANG) 
dev_split = data["dev"]


# ==========================================
# Core Functions
# ==========================================

class PaperExtraction(BaseModel):
    title: str = Field(description="The exact title of the referenced research paper.")
    authors: str = Field(description="The authors of the paper.")
    keyterms: list[str] = Field(description="A list of 3 to 5 specific, single-word keyterms mentioned.")

def get_llm_extraction(tweet_text):
    prompt = (
        "Read the following tweet and extract the referenced research paper. "
        "If you dont know which research paper, make an informed guess. "
        "Return the title, authors, and a list of keyterms. "
        "\n\n"
        f"Tweet: {tweet_text}"
    )
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": PaperExtraction,      
                "temperature": 0.0                        
            }
        )
        
        result: PaperExtraction = response.parsed
        return result.title, result.authors, result.keyterms
        
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

def get_top_5_matches(extracted_title, search_terms):
    if not extracted_title:
        return []
        
    extracted_title_lower = extracted_title.lower()
    
    # Cutoff lowered to 0.4. If it is 1, it requires an exact match and defeats the purpose of difflib.
    matches = difflib.get_close_matches(extracted_title_lower, collection_titles, n=5, cutoff=0.4)
    
    if not matches:
        return []
        
    candidates = []
    for match_title in matches:
        pubkey = title_to_pubkey[match_title]
        abstract = pubkey_to_abstract.get(pubkey, "")
        kw_matches = count_keyterm_matches(search_terms, abstract)
        
        # Calculate sequence similarity (Python's equivalent to normalized edit distance)
        sim_ratio = difflib.SequenceMatcher(None, extracted_title_lower, match_title).ratio()
        
        candidates.append({
            "title": pubkey_to_title[pubkey],
            "pubkey": pubkey,
            "sim_ratio": sim_ratio,
            "kw_matches": kw_matches
        })
        
    # Sort strictly by similarity ratio (edit distance) descending
    candidates.sort(key=lambda x: x["sim_ratio"], reverse=True)
    
    return candidates

def evaluate_system(tweets_dataset, num_samples):
    correct_matches = 0
    total_processed = 0
    limit = min(num_samples, len(tweets_dataset["text"]))
    
    print(f"\nStarting evaluation on {limit} tweets using {MODEL_NAME}...\n")
    
    for i in range(limit):
        tweet_text = tweets_dataset["text"][i] 
        true_pubkey = tweets_dataset["pubkey"][i]
        
        true_title = pubkey_to_title.get(true_pubkey, "NOT_FOUND")
        true_abstract = pubkey_to_abstract.get(true_pubkey, "")
        
        print(f"[{i+1}/{limit}] Tweet: {tweet_text}...")
        
        # Step 1: LLM Extraction
        extracted_title, extracted_authors, raw_keyterms = get_llm_extraction(tweet_text)
        
        processed_terms = set()
        for term in raw_keyterms:
            for word in term.split():
                clean_word = word.strip('.,!?()"\'').lower()
                if len(clean_word) > 1: 
                    processed_terms.add(clean_word)
                    
        print(f"   LLM Title Guessed : '{extracted_title}'")
        print(f"   Search Terms      : {list(processed_terms)}")
        
        # Step 2: Get Top 5 Sorted by Edit Distance (Similarity Ratio)
        candidates = get_top_5_matches(extracted_title, processed_terms)
        true_matches = count_keyterm_matches(processed_terms, true_abstract)
        
        predicted_pubkey = None
        
        if candidates:
            print("\n   Top 5 Candidates (Sorted by String Similarity):")
            for rank, candidate in enumerate(candidates, 1):
                marker = "[CORRECT]" if candidate["pubkey"] == true_pubkey else "[       ]"
                title_preview = candidate["title"][:60] + ("..." if len(candidate["title"]) > 60 else "")
                
                print(f"   {rank}. {marker} Sim: {candidate['sim_ratio']:.4f} | KW: {candidate['kw_matches']} | Title: '{title_preview}'")
            
            # The predicted pubkey is the #1 item based on string similarity
            predicted_pubkey = candidates[0]["pubkey"]
            print(f"\n   Selected Paper         : '{candidates[0]['title']}'")
            print(f"   Term hits in SELECTED  : {candidates[0]['kw_matches']} / {len(processed_terms)}")
        else:
            print(f"\n   Selected Paper         : None")

        print(f"   Term hits in TRUE abst : {true_matches} / {len(processed_terms)}")
        
        # Step 3: Evaluate
        if predicted_pubkey == true_pubkey:
            print("   Result: ✅ CORRECT PAPER IDENTIFIED\n")
            correct_matches += 1
        else:
            print(f"   True Title Was         : '{true_title}'")
            print("   Result: ❌ MISMATCH\n")
            
        print("-" * 50)
        total_processed += 1
        
    accuracy = (correct_matches / total_processed) * 100 if total_processed > 0 else 0
    
    print("="*40)
    print("      FINAL EVALUATION RESULTS")
    print("="*40)
    print(f"Model Used:      {MODEL_NAME}")
    print(f"Total Processed: {total_processed}")
    print(f"Correct Matches: {correct_matches}")
    print(f"Accuracy:        {accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate_system(dev_split, NUM_TWEETS_TO_EVALUATE)