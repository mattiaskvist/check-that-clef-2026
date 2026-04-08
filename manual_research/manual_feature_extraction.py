import spacy
from rapidfuzz import fuzz
from tqdm import tqdm
from datasets import load_dataset
import country_converter as coco
from geotext import GeoText
import logging

# Mute the country_converter spam
logging.getLogger('country_converter').setLevel(logging.ERROR)

# ==========================================
# 1. INITIALIZATION
# ==========================================
print("Loading NLP models and converters...")
nlp = spacy.load("en_core_web_sm")
cc = coco.CountryConverter()
# ==========================================
# 1. INITIALIZATION
# ==========================================

# ==========================================
# 2. MODULAR EXTRACTORS
# ==========================================

def extract_persons(text):
    """Extracts people's names from the text."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

def extract_locations(text):
    """Extracts countries, cities, and states."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]

# ==========================================
# 3. NORMALIZATION & MATCHING
# ==========================================

def normalize_location(entity):
    """
    Expands an extracted location into a list of search terms.
    For example: 
    - "USA" -> ["usa", "united states"]
    - "Melbourne" -> ["melbourne", "australia"]
    """
    search_terms = [entity.lower().strip()]
    
    # 1. Try to normalize as a country (e.g., 'USA' -> 'United States')
    country_name = cc.convert(names=entity, to='name_short', not_found=None)
    if country_name:
        search_terms.append(country_name.lower().strip())
        
    # 2. Try to map a city to a country (e.g., 'Melbourne' -> 'Australia')
    geo = GeoText(entity.title()) 
    if geo.country_mentions: 
        # Grab the ISO code with the highest frequency (the first key)
        iso_code = list(geo.country_mentions.keys())[0] 
        
        # Convert the ISO code ('AU') to the full country name ('Australia')
        parent_country = cc.convert(names=iso_code, to='name_short', not_found=None)
        if parent_country:
            search_terms.append(parent_country.lower().strip())
            
    return list(set(search_terms))

def is_fuzzy_match(extracted_entities, target_text, entity_type="person", threshold=80):
    """
    Checks if any of the extracted entities fuzzily match the target text.
    """
    if not extracted_entities or not target_text:
        return False
        
    target_lower = target_text.lower()
        
    for entity in extracted_entities:
        
        # If it's a location, expand it into synonyms and parent countries
        if entity_type == "location":
            search_terms = normalize_location(entity)
        else:
            # If it's a person, just lowercase it
            search_terms = [entity.lower().strip()]
        
        # Check if ANY of the terms match the paper's text
        for term in search_terms:
            score = fuzz.token_set_ratio(term, target_lower)
            if score >= threshold:
                return True
                
    return False

# ==========================================
# 4. EVALUATION PIPELINE
# ==========================================

def evaluate_extraction(tweets, collection_dict):
    """
    Evaluates how often we extract an entity AND find it in the ground-truth paper.
    """
    metrics = {
        "persons": {"extracted_count": 0, "matched_in_paper": 0},
        "locations": {"extracted_count": 0, "matched_in_paper": 0}
    }
    
    print("Evaluating extraction and matching...")
    
    for tweet in tqdm(tweets):
        tweet_text = tweet["text"]
        label_pubkey = tweet["pubkey"]
        
        # Get the ground truth paper
        ground_truth_paper = collection_dict.get(label_pubkey)
        if not ground_truth_paper:
            continue
            
        # Combine all paper fields to search within
        paper_full_text = (
            str(ground_truth_paper["title"]) + " " + 
            str(ground_truth_paper["abstract"]) + " " + 
            str(ground_truth_paper["authors"]) + " " + 
            str(ground_truth_paper["venue"])
        )
        
        # Run Extractors
        extracted_persons = extract_persons(tweet_text)
        extracted_locations = extract_locations(tweet_text)
        
        # Evaluate Persons
        if extracted_persons:
            metrics["persons"]["extracted_count"] += 1
            if is_fuzzy_match(extracted_persons, paper_full_text, entity_type="person"):
                metrics["persons"]["matched_in_paper"] += 1
                
        # Evaluate Locations
        if extracted_locations:
            metrics["locations"]["extracted_count"] += 1
            if is_fuzzy_match(extracted_locations, paper_full_text, entity_type="location"):
                metrics["locations"]["matched_in_paper"] += 1

    return metrics

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Loading datasets...")
    collection_raw = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "collection")["collection"]
    data_raw = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", "en")["train"]
    
    collection_records = collection_raw.to_list()
    tweets = data_raw.to_list()

    print("Building collection dictionary for fast lookups...")
    # Convert to a dictionary for O(1) lookups
    collection_dict = {row["pubkey"]: row for row in collection_records}

    # Run the evaluation
    results = evaluate_extraction(tweets, collection_dict)

    # Print summary
    print("\n===== EXTRACTION EVALUATION RESULTS =====")
    for feature, stats in results.items():
        extracted = stats["extracted_count"]
        matched = stats["matched_in_paper"]
        
        success_rate = (matched / extracted * 100) if extracted > 0 else 0
        
        print(f"Feature: {feature.upper()}")
        print(f"  - Found in {extracted}/{len(tweets)} tweets.")
        print(f"  - When found, successfully matched to paper {matched} times ({success_rate:.1f}% match rate).")
        print("-" * 40)

if __name__ == "__main__":
    main()