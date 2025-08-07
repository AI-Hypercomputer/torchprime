"""Utilities for preparing Hugging Face assets (models and tokenizers) for GCS."""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

TOKENIZER_PATTERNS = [
  "tokenizer.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  "*.model",  # For sentencepiece tokenizers
  "vocab.txt",  # For WordPiece/BERT tokenizers
  "merges.txt",  # For BPE tokenizers
]

MODEL_PATTERNS = [
  "*.safetensors*",
  "config.json",
  "generation_config.json",
]


def _upload_directory_to_gcs(local_path: Path, gcs_path: str):
  """Uploads the contents of a local directory to GCS using gsutil."""
  if not gcs_path.startswith("gs://"):
    raise ValueError("GCS path must start with gs://")

  logger.info(f"Uploading contents of '{local_path}' to '{gcs_path}'...")
  # Using gsutil for efficient, parallel uploads.
  # The '/*' at the end of local_path ensures the contents are copied, not the directory itself.
  command = ["gsutil", "-m", "cp", "-r", f"{str(local_path).rstrip('/')}/*", gcs_path]
  try:
    subprocess.run(command, check=True, capture_output=True, text=True)
    logger.info(f"Successfully uploaded assets to {gcs_path}.")
  except subprocess.CalledProcessError as e:
    logger.error(f"Failed to upload to GCS. Error: {e.stderr}")
    raise


def save_hf_model_files_to_gcs(
  repo_id: str,
  gcs_path: str,
  file_type: str,
  temp_dir: str | None = None,
):
  """Downloads model and tokenizer files from a Hugging Face repo and uploads to GCS."""
  allow_patterns = []
  if file_type in ("tokenizer", "all"):
    allow_patterns.extend(TOKENIZER_PATTERNS)
  if file_type in ("model", "all"):
    allow_patterns.extend(MODEL_PATTERNS)

  if not allow_patterns:
    raise ValueError("file_type must be one of 'tokenizer', 'model', or 'all'")

  with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir:
    logger.info(f"Created temporary directory: {tmpdir}")

    logger.info(f"Downloading files for '{repo_id}' with patterns: {allow_patterns}")
    snapshot_path = snapshot_download(
      repo_id=repo_id,
      cache_dir=str(tmpdir),
      token=os.environ.get("HF_TOKEN"),
      allow_patterns=allow_patterns,
      ignore_patterns=["*.bin*"],  # Avoid large pytorch_model.bin files
    )

    logger.info(f"Files for '{repo_id}' downloaded locally to '{snapshot_path}'.")

    # Upload the directory contents to the specified GCS path.
    _upload_directory_to_gcs(Path(snapshot_path), gcs_path)
