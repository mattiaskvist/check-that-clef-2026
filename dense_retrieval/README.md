## How to Execute the Pipeline
Because Modal runs in the cloud, your remote GPU container won't automatically have access to your local JSON dataset. You will need to upload your data to the Modal Volume before initiating the training.

Step A: Upload your training data to the Volume
Use the Modal CLI to push your your_hard_negative_triplets.json file into the remote volume:

```bash
uv run modal volume put model-weights-vol your_hard_negative_triplets.json /your_hard_negative_triplets.json
```

Step B: Run the Training Script

```bash
uv run modal run train_bge_modal.py
```

Modal will dynamically provision the container, attach the GPU, mount the volume containing your dataset, and execute the training loop.

Step C: Download your Fine-Tuned Model
Once training finishes, your LoRA adapter weights will be safely stored in the cloud. You can download the completed model folder back to your local machine using the CLI:

```bash
uv run modal volume get model-weights-vol /bge-m3-finetuned./my-local-bge-m3-model
```