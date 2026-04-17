# Training (`src/clef_training`)

These scripts handle dense-retrieval training/inference workflows.

## How to execute training on Modal

Because Modal runs in the cloud, your remote GPU container won't automatically have access to your local JSON dataset. Upload your dataset to the Modal volume before training.

Step A: Upload your training data to the Volume
Use the Modal CLI to push your your_hard_negative_triplets.json file into the remote volume:

```bash
uv run modal volume create clef-vol
uv run modal volume put clef-vol hard_negative_triplets.json /hard_negative_triplets.json
```

Step B: Run the Training Script in the Modal Container

```bash
uv run modal run -d -m clef_training.train_bge_modal
```

Modal will dynamically provision the container, attach the GPU, mount the volume containing your dataset, and execute the training loop.

Step C: Download your Fine-Tuned Model
Once training finishes, your LoRA adapter weights will be safely stored in the cloud. You can download the completed model folder back to your local machine using the CLI:

```bash
uv run modal volume get clef-vol /bge-m3-finetuned./my-local-bge-m3-model
```
