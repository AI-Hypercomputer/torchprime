"""
This script downloads specified model and tokenizer files from the Hugging Face Hub
and uploads them to a Google Cloud Storage (GCS) bucket.

It is designed to prepare files required for training runs.

Usage:
  python torchprime/tools/prepare_gcs_model_files.py gs://your-bucket/your-path [--temp-dir /path/to/temp]
"""

import os
import sys

from torchprime.launcher import save_hf_tokenizer_and_model

# --- Configuration ---
# List of models and the specific files to download for each.
# file_type can be 'tokenizer', 'model', or 'all'.
FILES_TO_PREPARE = [
  {
    "repo_id": "meta-llama/Meta-Llama-3-8B",
    "gcs_dir_name": "meta-llama-3-8b",
    "file_type": "all",  # Download model weights, config, and tokenizer
  },
  {
    "repo_id": "meta-llama/Meta-Llama-3.1-405B",
    "gcs_dir_name": "meta-llama-3.1-405b",
    "file_type": "tokenizer",
  },
  {
    "repo_id": "meta-llama/Llama-4-Scout-17B-16E",
    "gcs_dir_name": "llama-4-scout-17b-16e",
    "file_type": "tokenizer",
  },
  {
    "repo_id": "mistralai/Mixtral-8x7B-v0.1",
    "gcs_dir_name": "mixtral-8x7b-v0.1",
    "file_type": "tokenizer",
  },
]


def main():
  """Downloads and uploads specified Hugging Face files to GCS."""
  if len(sys.argv) < 2 or not sys.argv[1].startswith("gs://"):
    print(
      f"Usage: python {sys.argv[0]} gs://your-bucket/your-path [--temp-dir /path/to/temp]",
      file=sys.stderr,
    )
    sys.exit(1)

  gcs_base_path = sys.argv[1]
  temp_dir = None
  if "--temp-dir" in sys.argv:
    try:
      idx = sys.argv.index("--temp-dir")
      temp_dir = sys.argv[idx + 1]
    except IndexError:
      print("Error: --temp-dir requires a path.", file=sys.stderr)
      sys.exit(1)

  if not os.environ.get("HF_TOKEN"):
    raise RuntimeError(
      "The HF_TOKEN environment variable is not set. "
      "Please run 'huggingface-cli login' or export your token."
    )

  print(f"--- Starting file preparation for GCS path: {gcs_base_path} ---")

  for i, file_info in enumerate(FILES_TO_PREPARE):
    repo_id, gcs_dir, file_type = (
      file_info["repo_id"],
      file_info["gcs_dir_name"],
      file_info["file_type"],
    )
    gcs_path = f"{gcs_base_path.rstrip('/')}/{gcs_dir}"
    print(f"\n[{i + 1}/{len(FILES_TO_PREPARE)}] Processing '{repo_id}'...")
    save_hf_tokenizer_and_model.save_hf_model_files_to_gcs(
      repo_id=repo_id, gcs_path=gcs_path, file_type=file_type, temp_dir=temp_dir
    )
    print(f"  -> Successfully saved '{file_type}' files for '{repo_id}' to {gcs_path}")

  print("\n--- File preparation complete. ---")


if __name__ == "__main__":
  main()
