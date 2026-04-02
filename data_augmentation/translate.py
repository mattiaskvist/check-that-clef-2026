import os
from datasets import load_dataset
from google import genai

# 1. Initialize the Gemini Client
# The client automatically picks up the GEMINI_API_KEY environment variable.
client = genai.Client()

lang = "en"  # Change to "de" or "fr" for German or French datasets respectively
split = "train"  # Change to "dev" for the development set

# 2. Load the dataset from Hugging Face
print("Loading dataset from Hugging Face...")
data = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", lang)
    
datasplit = data[split]

# 3. Access the text field
# The dataset contains a 'text' column. Let's grab the very first entry.
original_text = datasplit[0]['text']

print("\n--- ORIGINAL ENGLISH TEXT (Snippet) ---")
print(original_text[:500] + "...\n")

# 4. Set up the translation prompt
# Providing clear instructions in the prompt ensures Gemini only outputs the translation
prompt = f"""
Translate the following English tweet about a scientific paper into French. 
Only return the translated text without any conversational filler.

Text to translate:
{original_text}
"""

# 5. Call the Gemini API to translate
print("Translating with Gemini...\n")
response = client.models.generate_content(
    model='gemini-3.1-flash-lite-preview',
    contents=prompt,
)

print("--- FRENCH TRANSLATION (Snippet) ---")
print(response.text[:500] + "...")