# check-that-clef-2026

## Quick start

```bash
brew install uv
uv sync

# Run evaluation pipeline
uv run python -m clef_pipeline.main

# Run Streamlit demo locally
uv run streamlit run src/clef_demo/clef_demo/streamlit_app.py

# To add or remove dependencies, use the following commands:
uv add <dependency>
uv remove <dependency>

# To run formatting and linting checks, use:
uv run ruff format
uv run ruff check

# Authenticate with HuggingFace
uv run hf auth login
```

## Code Layout

- `src/clef_pipeline/` — retrieval pipeline, evaluation orchestration, metrics, submission export
- `src/clef_demo/` — Streamlit UI and Modal web hosting entrypoint
- `src/clef_training/` — dense-retrieval training and helper scripts
- `tests/` — regression and modular pipeline tests

## Modal commands

```bash
# Evaluation pipeline
uv run modal run -m clef_pipeline.main --split dev

# Streamlit demo on Modal
uv run modal serve -m clef_demo.modal_app
```

## Submission Guidelines

https://www.codabench.org/competitions/15611/#/pages-tab

Each team must create only one account in CodaBench and submit their predictions exclusively through that account.
Make sure your account name matches that used during CLEF registration.
The last valid submission will be considered as the final submission!
The submission file predictions.zip must include your predictions.
Inside the predictions.zip file include .tsv files (TSV, not CSV) for language-specific predictions.
The .tsv files must be named predictions_{lang}.tsv where {lang} is either en, de, or fr.
The predictions.zip file must include the prediction .tsv files for the languages you want to participate in.
The predictions_{lang}.tsv must contain the following columns: "index", "preds", where "index" is the index of the post and "preds" contains an array of the top5 predicted pubkeys (in descending order: [pred1, pred2, pred3, pred4, pred5], where pred1 is the pubkey of the highest ranked publication from the collection set)
You are allowed to submit max 50 submissions per day per team.
