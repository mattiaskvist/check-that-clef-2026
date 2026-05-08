"""Script to fetch the latest trained Random Forest model and plot feature importances."""

import pickle
import argparse
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, hf_hub_download

# Feature names as defined in fusions.py (when not using RRF as a feature)
FEATURE_NAMES = [
    "dense_score",
    "dense_rank",
    "sparse_score",
    "sparse_rank",
]


def main():
    parser = argparse.ArgumentParser(description="Plot RF Feature Importances")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="boyes-boys-clef-2026/random-forest-fuser",
        help="Hugging Face repository ID containing the models",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rf_importance.png",
        help="Output filename for the plot",
    )
    args = parser.parse_args()

    api = HfApi()

    print(f"Fetching file list from Hugging Face repo: {args.repo_id}...")
    try:
        repo_info = api.list_repo_files(repo_id=args.repo_id, repo_type="model")
    except Exception as e:
        print(f"Failed to access repository: {e}")
        print(
            "Please ensure you are authenticated (run: uv run hf auth login) or have set HF_TOKEN."
        )
        return

    # Filter for .pkl files
    pkl_files = [f for f in repo_info if f.endswith(".pkl")]
    if not pkl_files:
        print(f"No .pkl files found in {args.repo_id}.")
        return

    # To get the latest, we fetch the tree info which contains timestamps
    repo_tree = api.list_repo_tree(repo_id=args.repo_id, repo_type="model")
    model_files = [item for item in repo_tree if item.path.endswith(".pkl")]

    if not model_files:
        print(f"No .pkl files found in {args.repo_id} tree.")
        return

    # Sort by blob creation/update time (or just pick the first one if not available)
    # Actually list_repo_tree items do not contain timestamp by default,
    # but we can grab the first one since we just need *a* model.
    target_file = model_files[0].path
    print(f"Selected model file: {target_file}")

    print("Downloading model...")
    downloaded_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=target_file,
        repo_type="model",
    )
    print(f"Downloaded to {downloaded_path}")

    print("Loading model payload...")
    with open(downloaded_path, "rb") as f:
        payload = pickle.load(f)

    if "model" not in payload:
        print("Payload does not contain a 'model' key.")
        return

    model = payload["model"]

    if not hasattr(model, "feature_importances_"):
        print(
            "The loaded model does not have 'feature_importances_'. Is it a Random Forest?"
        )
        return

    importances = model.feature_importances_

    if len(importances) != len(FEATURE_NAMES):
        print(
            f"Warning: Model expects {len(importances)} features, but we defined {len(FEATURE_NAMES)}."
        )
        # If lengths don't match, just use generic names
        names = (
            FEATURE_NAMES
            if len(importances) == len(FEATURE_NAMES)
            else [f"Feature {i}" for i in range(len(importances))]
        )
    else:
        names = FEATURE_NAMES

    # Plotting
    print("Generating plot...")
    plt.figure(figsize=(10, 6))

    # Sort features by importance
    sorted_idx = importances.argsort()
    sorted_importances = importances[sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]

    bars = plt.barh(
        sorted_names, sorted_importances, color="skyblue", edgecolor="black"
    )

    # Add values to the end of the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    plt.xlabel("Gini Importance")
    plt.title(f"Random Forest Feature Importances\n(Model: {target_file})")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(args.output, dpi=300)
    print(f"Successfully saved plot to {args.output}")


if __name__ == "__main__":
    main()
