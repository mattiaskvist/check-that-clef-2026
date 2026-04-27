# BGE-M3 Fine-Tuning

`src/clef_training` contains the scripts used to mine hard negatives and train a
LoRA adapter for BGE-M3 on the CheckThat source retrieval task.

The workflow is separate from the evaluation pipeline. It produces model
artifacts that can later be referenced by the `bge-m3` retriever in
`clef_pipeline`.

## Files

- `clef_training/hard_negative_mining.py`: builds JSONL triplets with
  `anchor`, `positive`, and `negative` fields.
- `clef_training/train_bge_modal.py`: trains a BGE-M3 LoRA adapter on Modal.
- `clef_training/inference.py`: local smoke test for loading a trained adapter
  and scoring a query against example documents.

## Mine Hard Negatives

Run from the repository root:

```bash
uv run python -m clef_training.hard_negative_mining
```

The script converts the labeled training queries into retrieval triplets for
contrastive dense-retriever training. Each triplet contains:

- `anchor`: the social media post or claim text.
- `positive`: the title and abstract of the known relevant paper.
- `negative`: the title and abstract of a lexically plausible but incorrect
  paper.

The mining process is intentionally simple and reproducible:

1. Load the CheckThat collection and the English, German, and French `train`
   query splits from Hugging Face.
2. Build a `pubkey -> "title\nabstract"` map for all collection documents.
3. Tokenize each collection document with lowercase whitespace splitting and
   build a BM25 index over those document texts.
4. For English queries, run BM25 with the English claim text and inspect the
   top 10 ranked collection documents.
5. Select the first high-ranking document whose `pubkey` is not the labeled
   positive `pubkey`; that document becomes the hard negative.
6. For German and French queries, first try to reuse the English query text
   that points to the same positive `pubkey`. This gives the lexical miner an
   English proxy query against an English scientific-paper collection. If no
   English proxy exists, the script falls back to the original German or French
   text.
7. Skip any query whose labeled positive paper is missing from the collection,
   or where no non-positive BM25 candidate is found in the inspected top
   results.

This writes:

```text
hard_negative_triplets.json
```

Despite the `.json` extension, the output is newline-delimited JSON: one triplet
object per line, not a single JSON array. Example:

```json
{"anchor": "claim text", "positive": "Paper title\nPaper abstract", "negative": "Other title\nOther abstract"}
```

The negatives are "hard" because BM25 considered them close to the query, but
they are still weak labels: the script only guarantees that the selected
negative is not the known positive `pubkey`. It does not prove that the
negative paper is scientifically unrelated. The generated file is ignored by
git because it is a local training artifact.

## Upload Training Data To Modal

Create the training volume and upload the triplets:

```bash
uv run modal volume create clef-vol
uv run modal volume put clef-vol hard_negative_triplets.json /hard_negative_triplets.json
```

Authenticate Modal and make the Hugging Face token available to the remote
container:

```bash
uv run modal setup
uv run modal secret create hf-token HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## Train On Modal

Launch the remote training job:

```bash
uv run modal run -d -m clef_training.train_bge_modal
```

The training app:

- loads `BAAI/bge-m3`;
- applies a LoRA configuration to the final transformer layers;
- trains on `/data/hard_negative_triplets.json`;
- evaluates against multilingual dev queries;
- saves checkpoints under `/data/bge-m3-checkthat-checkpoints`; and
- saves the selected final adapter under `/data/bge-m3-finetuned`.

The Modal function currently requests an `A100-40GB` GPU and has a 12-hour
timeout.

## Download The Fine-Tuned Model

After training completes, download the final model directory:

```bash
uv run modal volume get clef-vol /bge-m3-finetuned ./bge-m3-finetuned
```

Keep downloaded checkpoints and model directories out of git.

## Inference Smoke Test

`clef_training/inference.py` demonstrates how to load the base BGE-M3 model,
inject a PEFT adapter, and compare a query against example documents:

```bash
uv run python -m clef_training.inference
```

The adapter id in that file must point to an accessible Hugging Face repository
or local adapter path. If the adapter repository is private, authenticate with
Hugging Face before running the smoke test.

## Model Visibility Notes

- Do not publish private adapter IDs unless the model repository is intended to
  be public.
- The pipeline registry currently uses
  `boyes-boys-clef-2026/bge-m3-checkthat-finetuned` for the `bge-m3` retriever.
- If the trained adapter is moved or renamed, update the registry constant and
  this README together.
