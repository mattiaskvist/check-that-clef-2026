import csv
import os

SUBMISSION_TOP_K = 5
ALLOWED_SPLITS = frozenset({"train", "dev", "test"})


def _format_preds(preds: list[str]) -> str:
    if len(preds) != SUBMISSION_TOP_K:
        raise ValueError(
            f"Each prediction row must contain exactly {SUBMISSION_TOP_K} pubkeys."
        )
    return str(preds)


def write_submission_tsv_files(
    predictions_by_lang: dict[str, list[dict[str, object]]],
    output_dir: str,
) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    written_paths = []

    for lang in sorted(predictions_by_lang):
        output_path = os.path.join(output_dir, f"predictions_{lang}.tsv")
        with open(output_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(["index", "preds"])
            for row in predictions_by_lang[lang]:
                preds = row.get("preds")
                if not isinstance(preds, list):
                    raise TypeError("'preds' must be a list of publication keys.")
                writer.writerow([row.get("index"), _format_preds(preds)])
        written_paths.append(output_path)

    return written_paths


def submission_volume_remote_dir(output_dir: str, cache_mount: str) -> str:
    output_dir_abs = os.path.abspath(output_dir)
    cache_mount_abs = os.path.abspath(cache_mount)

    prefix = f"{cache_mount_abs}{os.sep}"
    if output_dir_abs == cache_mount_abs:
        return "/"
    if not output_dir_abs.startswith(prefix):
        raise ValueError(f"Output directory must be under cache mount: {cache_mount_abs}")

    relative_path = os.path.relpath(output_dir_abs, cache_mount_abs)
    return f"/{relative_path.replace(os.sep, '/')}"


def modal_volume_download_command(
    volume_name: str, remote_dir: str, local_destination: str
) -> str:
    return f"uv run modal volume get {volume_name} {remote_dir} {local_destination}"


def normalize_split(split: str) -> str:
    if split not in ALLOWED_SPLITS:
        raise ValueError("split must be one of: train, dev, test.")
    return split
