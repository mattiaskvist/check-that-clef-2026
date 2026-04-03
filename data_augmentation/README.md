# Cross-Lingual Data Augmentation and Alignment

The stark quantitative imbalance between the 15,699 English training pairs and the approximately 1,500 German and French pairs presents a severe, systemic risk of catastrophic underfitting for the European languages. To rectify this asymmetry, the robust English training corpus is synthetically translated into German and French utilizing high-fidelity generative AI. This synthetic data generation effectively scales the low-resource splits, allowing representation models fine-tuned on the data to learn the specific semantic nuances of scientific claim retrieval in DE and FR utilizing the rich structural diversity of the massive English dataset, rather than overfitting on a tiny subset of 1,500 examples.

## Purpose

The primary goal of this augmentation pipeline is to create high-quality synthetic training data for German and French by translating the existing English training set. This approach ensures that the models trained on these augmented datasets can learn from a much larger and more diverse set of examples, improving their performance on the retrieval task for scientific claims in all three languages.

## Data Augmentation Pipeline

Instead of relying on traditional Neural Machine Translation (NMT) APIs, this project utilizes the **Gemini API** (`gemini-3.1-flash-lite-preview`) combined with **Structured Outputs (Pydantic)** to generate dual translations simultaneously and reliably.

1. **Access and Load the Datasets**: The pipeline uses the Hugging Face `datasets` library to automatically download and cache the source data (`sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims`). 
2. **Structured Translation Infrastructure**: By leveraging Pydantic schemas, we force the Gemini model to return a strict JSON payload containing both the German and French translations in a single API call. This eliminates conversational filler and drastically speeds up the pipeline.
3. **Execute the Translation Loop**: The script iterates through the English training data, translating only the `text` column (the informal social media posts).
4. **Preserve the Ground Truth Mappings**: As the text is translated, the pipeline strictly maintains the exact mapping to the original `pubkey`. The `pubkey` is the unique identifier that links the synthetic post to the correct target paper in the shared retrieval pool.
5. **Merge and Save**: The newly translated synthetic German and French datasets are automatically appended to the official existing splits, normalized with a strict three-column structure (`index`, `pubkey`, `text`), and exported locally as `de_train.json` and `fr_train.json`.

## Usage

To execute the translation augmentation pipeline locally, follow these steps:

**1. Configure Environment Variables**

Create a `.env` file in the root directory of the script and add your Gemini API key, following the .env.template format:

```txt
GEMINI_API_KEY=your_gemini_api_key_here
```

**2. Run the Script**

Execute the translation script:

```bash
uv run data_augmentation/translate.py
```

This will trigger the entire augmentation process, which may take some time depending on the size of the dataset and the response times of the Gemini API. The final translated datasets will be saved in the `data_augmentation` directory as `de_train.json` and `fr_train.json`. It translates about 30 tweets per second, so the entire English training set of 15,699 rows should complete in approximately 8-9 minutes.
