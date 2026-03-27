import os
import difflib
from datasets import load_dataset
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ==========================================
# Configuration
# ==========================================
# We use Gemini 2.5 Flash, which is incredibly fast and cheap/free for this volume
MODEL_NAME = 'gemini-2.5-flash' 
LANG = "en"
NUM_TWEETS_TO_EVALUATE = 15

load_dotenv()

# The Google SDK automatically looks for the GEMINI_API_KEY environment variable

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

# Define our strict output structure using Pydantic.
# Adding descriptions here actually helps Gemini understand what to extract!
class PaperExtraction(BaseModel):
    title: str = Field(description="The exact title of the referenced research paper.")
    authors: str = Field(description="The authors of the paper.")
    keyterms: list[str] = Field(description="A list of 3 to 5 specific, single-word keyterms mentioned.")

def get_llm_extraction(tweet_text):
    prompt = (
        "Read the following tweet and extract the referenced research paper. "
        "If you dont know which research paper, make an informed guess"
        "Return the title, authors, and a list of keyterms. "
        "\n\n"
        f"Tweet: {tweet_text}"
    )
    
    try:
        # Generate content with strict schema enforcement
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "response_mime_type": "application/json", # Forces JSON mode
                "response_schema": PaperExtraction,       # Enforces our exact Pydantic schema
                "temperature": 0.0                        # Zero temp for strict data extraction
            }
        )
        
        # The SDK automatically parses the JSON back into our PaperExtraction Python object!
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

def find_best_match(extracted_title, search_terms):
    if not extracted_title:
        return None, None, 0
        
    extracted_title_lower = extracted_title.lower()
    
    matches = difflib.get_close_matches(extracted_title_lower, collection_titles, n=5, cutoff=0.4)
    
    if not matches:
        return None, None, 0
        
    best_match_title = matches[0]
    best_pubkey = title_to_pubkey[best_match_title]
    best_abstract = pubkey_to_abstract.get(best_pubkey, "")
    max_matches = count_keyterm_matches(search_terms, best_abstract)
    
    for match in matches[1:]:
        pubkey = title_to_pubkey[match]
        abstract = pubkey_to_abstract.get(pubkey, "")
        kw_matches = count_keyterm_matches(search_terms, abstract)
        
        if kw_matches > max_matches:
            max_matches = kw_matches
            best_match_title = match
            best_pubkey = pubkey
            
    return best_match_title, best_pubkey, max_matches

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
        
        print(f"[{i+1}/{limit}] Tweet: {tweet_text[:100]}...")
        
        # Step 1: Gemini Extraction
        extracted_title, extracted_authors, raw_keyterms = get_llm_extraction(tweet_text)
        
        processed_terms = set()
        for term in raw_keyterms:
            for word in term.split():
                clean_word = word.strip('.,!?()"\'').lower()
                if len(clean_word) > 1: 
                    processed_terms.add(clean_word)
                    
        print(f"   Gemini Title Guessed : '{extracted_title}'")
        print(f"   Search Terms         : {list(processed_terms)}")
        
        # Step 2: Fuzzy Match & Re-ranking
        matched_title_lower, predicted_pubkey, predicted_matches = find_best_match(extracted_title, processed_terms)
        
        true_matches = count_keyterm_matches(processed_terms, true_abstract)
        
        if predicted_pubkey:
            matched_title_display = pubkey_to_title[predicted_pubkey]
            print(f"   Selected Paper       : '{matched_title_display}'")
        else:
            print(f"   Selected Paper       : None")

        print(f"   Term hits in TRUE abstract     : {true_matches} / {len(processed_terms)}")
        if predicted_pubkey:
            print(f"   Term hits in SELECTED abstract : {predicted_matches} / {len(processed_terms)}")
        
        # Step 3: Evaluate
        if predicted_pubkey == true_pubkey:
            print("   Result: ✅ CORRECT PAPER IDENTIFIED\n")
            correct_matches += 1
        else:
            print(f"   True Title Was       : '{true_title}'")
            print("   Result: ❌ MISMATCH\n")
            
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