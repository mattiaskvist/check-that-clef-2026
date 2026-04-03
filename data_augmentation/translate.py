"""
Dataset Translation Pipeline

This script loads a multilingual claim retrieval dataset from Hugging Face,
extracts a subset of the English training split, and translates the text into
both German and French using the Gemini API. The translations are forced into a
structured JSON format using Pydantic. Finally, the translated records are
merged with the existing German and French datasets and saved locally as JSON files.
"""

import asyncio
import json
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from tqdm.asyncio import tqdm

load_dotenv()

MODEL_NAME = "gemini-3.1-flash-lite-preview"
DATASET_NAME = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent

# --- Concurrency Configuration ---
# 50 concurrent requests x ~1 sec per request = ~3,000 Requests Per Minute
CONCURRENCY_LIMIT = 50


class Translations(BaseModel):
    """
    Pydantic schema to enforce structured JSON output from the Gemini API.

    Attributes:
        german (str): The translated German text.
        french (str): The translated French text.
    """

    german: str
    french: str


# Initialize the Gemini client (expects GEMINI_API_KEY environment variable)
client = genai.Client()


def translate_text(text: str) -> Translations:
    """
    Translates a given English text into German and French using the Gemini API.

    This function utilizes Gemini's structured output capabilities to guarantee
    the response matches the `Translations` Pydantic model.

    Args:
        text (str): The original English text to be translated.

    Returns:
        Translations: A parsed Pydantic object containing the `german` and
                      `french` translation strings.
    """
    prompt = f"""Translate the following English text into German and French.
Only return the translated texts without any conversational filler.

Text to translate:
{text}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": Translations,
            "temperature": 0.0,
        },
    )

    return response.parsed


async def translate_text_async(text: str, semaphore: asyncio.Semaphore) -> Translations:
    """
    Async function that requests translations while respecting the concurrency limit.
    """
    prompt = f"""Translate the following English text into German and French.
Only return the translated texts without any conversational filler.

Text to translate:
{text}
"""
    # The semaphore ensures we don't exceed our allowed concurrent requests
    async with semaphore:
        # Note the use of client.aio for asynchronous calls
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": Translations,
                "temperature": 0.0,
            },
        )
        return response.parsed


def normalize_rows(rows: list[dict]) -> list[dict]:
    """
    Standardizes a list of dataset rows by injecting an absolute index.

    Args:
        rows (list[dict]): A list of dictionaries representing dataset rows.

    Returns:
        list[dict]: The updated list of rows, where each row now includes an
                    'index' key corresponding to its position in the list.
    """
    return [
        {
            "index": position,
            "pubkey": row["pubkey"],
            "text": row["text"],
        }
        for position, row in enumerate(rows)
    ]


async def process_row(row: dict, semaphore: asyncio.Semaphore) -> tuple:
    """Wrapper to map pubkeys to their async translation results."""
    try:
        translated_text = await translate_text_async(row["text"], semaphore)
        return row["pubkey"], translated_text, None
    except Exception as e:
        return row["pubkey"], None, e


async def translate_split(source_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Processes a list of source rows, translating the text of each row into
    German and French.

    Args:
        source_rows (list[dict]): A list of dictionaries containing the source
                                  English data. Expected to have 'pubkey' and
                                  'text' keys.

    Returns:
        tuple[list[dict], list[dict]]: A tuple containing two lists. The first
                                       is the list of translated German rows,
                                       and the second is the French rows.
    """
    german = []
    french = []

    # Initialize the concurrency bouncer
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # Create a list of async tasks for the entire dataset
    tasks = [process_row(row, semaphore) for row in source_rows]

    try:
        # asyncio.as_completed yields tasks as soon as they finish, letting us update the progress bar in real-time
        for completed_task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Translating (Async)...",
            unit="row",
            disable=not sys.stdout.isatty(),
        ):
            pubkey, trans, error = await completed_task

            if error:
                # If a specific row fails, we log it but don't crash the whole pipeline
                print(f"\n[!] Failed to translate pubkey {pubkey}: {error}")
                continue

            if trans:
                german.append({"pubkey": pubkey, "text": trans.german})
                french.append({"pubkey": pubkey, "text": trans.french})

    except (Exception, KeyboardInterrupt) as e:
        # Catch any API errors, network failures, or a manual Ctrl+C
        print(f"\n[!] Translation interrupted: {e}")
        print(f"[!] Salvaging {len(german)} successfully translated rows...")

    return german, french


def merge_with_existing_split(
    existing_rows: list[dict], translated_rows: list[dict]
) -> list[dict]:
    """
    Combines newly translated rows with the existing dataset split and normalizes
    the resulting dataset.

    Args:
        existing_rows (list[dict]): The pre-existing rows for the target language.
        translated_rows (list[dict]): The newly translated rows to append.

    Returns:
        list[dict]: A single, normalized list containing both existing and new rows.
    """
    merged_rows = []

    for row in existing_rows:
        merged_rows.append({"pubkey": row["pubkey"], "text": row["text"]})

    merged_rows.extend(translated_rows)

    return normalize_rows(merged_rows)


def write_json(path: Path, rows: list[dict]) -> None:
    """
    Writes a list of dictionaries to a JSON file, creating parent directories
    if necessary.

    Args:
        path (Path): The file path where the JSON data should be saved.
        rows (list[dict]): The dataset rows to serialize and save.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


async def main() -> None:
    """
    Main execution block. Loads datasets, triggers the translation pipeline
    on a randomized subset, merges the results, and writes out the final JSON files.
    """
    print("Loading dataset from Hugging Face...")

    # Load all required language splits
    english_data = load_dataset(DATASET_NAME, "en")
    german_data = load_dataset(DATASET_NAME, "de")
    french_data = load_dataset(DATASET_NAME, "fr")

    english_train = english_data["train"]
    # Skip the first x rows that you successfully translated last time but it crashed
    # english_train = english_data["train"].select(range(3982, len(english_data["train"])))
    german_train = german_data["train"]
    french_train = french_data["train"]

    print(
        f"Translating {len(english_train)} English training rows into German and French..."
    )
    translated_german, translated_french = await translate_split(english_train)

    # Merge translations with the original language datasets
    merged_german = merge_with_existing_split(german_train, translated_german)
    merged_french = merge_with_existing_split(french_train, translated_french)

    german_output = OUTPUT_DIRECTORY / "de_train.json"
    french_output = OUTPUT_DIRECTORY / "fr_train.json"

    print(f"Writing merged German training set to {german_output}...")
    write_json(german_output, merged_german)

    print(f"Writing merged French training set to {french_output}...")
    write_json(french_output, merged_french)

    await client.aio.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
