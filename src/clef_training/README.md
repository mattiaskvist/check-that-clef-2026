# Dense Retriever Fine-Tuning

`src/clef_training` contains the scripts used to mine hard negatives and train
LoRA adapters for dense retrievers on the CheckThat source retrieval task.

The workflow is separate from the evaluation pipeline. It produces model
artifacts that can later be referenced by retrievers in `clef_pipeline`.

## Files

- `clef_training/hard_negative_mining.py`: builds JSONL triplets with
  `anchor`, `positive`, and `negative` fields.
- `clef_training/train_bge_modal.py`: trains a BGE-M3 LoRA adapter on Modal.
- `clef_training/train_harrier_en_modal.py`: mines English Harrier false
  positives and trains a Harrier 27B LoRA adapter on Modal.
- `clef_training/inference.py`: local smoke test for loading a trained adapter
  and scoring a query against example documents.

## Harrier 27B English Fine-Tuning

`train_harrier_en_modal.py` is the English-only Harrier workflow. It has two
remote stages:

1. Mine hard negatives from current `microsoft/harrier-oss-v1-27b` retrieval
   mistakes on the English train split.
2. Train a PEFT LoRA adapter on those triplets with English dev IR evaluation.

Run both stages:

```bash
uv run modal run -d -m clef_training.train_harrier_en_modal \
  --mode all \
  --negatives-per-query 4 \
  --search-top-k 100
```

Run only mining:

```bash
uv run modal run -d -m clef_training.train_harrier_en_modal --mode mine
```

Run only training after triplets already exist in the Modal volume:

```bash
uv run modal run -d -m clef_training.train_harrier_en_modal --mode train
```

Training checkpoints are saved frequently by default:

```text
--save-steps 100
--eval-steps 1000
--save-total-limit 12
--resume-from-checkpoint auto
```

`auto` resumes from the newest `/data/harrier-27b-en-checkpoints/checkpoint-*`
directory. Each checkpoint save also commits the Modal volume, so a later
`--mode train` run can resume even after a container restart or interrupted job.
Use `--resume-from-checkpoint none` to force a fresh run.
The default separates checkpointing and evaluation for speed: recovery
checkpoints are frequent, while expensive English dev IR evaluation runs less
often. If you want the Trainer to reload the best English dev `ndcg@10`
checkpoint automatically, set `--eval-steps` equal to `--save-steps`.

Mining is recoverable too. The miner:

- caches Harrier document embeddings at `/data/harrier_en_document_embeddings.pt`;
- appends to `/data/harrier_en_hard_negatives.jsonl` when `--resume-mining` is
  enabled;
- skips query pubkeys already present in the JSONL; and
- commits the Modal volume every few mined batches.

Request two B200 GPUs for training:

```bash
uv run modal run -d -m clef_training.train_harrier_en_modal \
  --mode train \
  --multi-gpu
```

This exposes two GPUs to the Trainer. It may speed up training through the
Trainer/Accelerate distributed data-parallel path. It deliberately does not use
`device_map="auto"` because that can split Gemma/Harrier layers across devices
and trigger cross-device tensor errors during the forward pass. Treat it as
experimental: each process loads a full model replica on one GPU, so if a full
replica does not fit per B200, a dedicated DeepSpeed/FSDP sharded setup is the
next step.

The default artifacts are written to the `clef-vol` Modal volume:

```text
/data/harrier_en_hard_negatives.jsonl
/data/harrier_en_document_embeddings.pt
/data/harrier-27b-en-checkpoints
/data/harrier-27b-en-lora
```

The training job uses:

- English train queries only;
- Harrier-mined false positives as negatives;
- `TripletLoss` with cosine distance;
- LoRA target modules set to `all-linear` by default; and
- English dev `ndcg@10` for best-checkpoint selection.

The Modal image pins `transformers==4.57.6`, matching Harrier's model config.
This avoids newer Gemma3 processor paths that can fail with a missing
`preprocessor_config.json` for this text-only embedding checkpoint.

After training, download or publish `/data/harrier-27b-en-lora`, then pass that
adapter path or Hugging Face id to `HarrierRetriever(..., lora_id=...)` for
evaluation.

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
6. For German and French queries, use an English proxy only for the BM25
   negative search when possible. The script builds a lookup from each English
   training `pubkey` to its English claim text. If a German or French training
   row has the same positive `pubkey`, BM25 searches with that English claim
   instead of the German or French text. This makes the lexical search more
   reliable because the collection documents are mostly English scientific titles and
   abstracts. The generated triplet still keeps the original German or French
   query as the `anchor`; the proxy is only used to choose a better hard
   negative. If no English query exists for that `pubkey`, the script falls
   back to searching with the original German or French query text.
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
