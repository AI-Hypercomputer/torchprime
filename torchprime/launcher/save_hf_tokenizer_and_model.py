"""Utilities for preparing Hugging Face assets (models and tokenizers) for GCS."""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

TOKENIZER_PATTERNS = (
  "tokenizer.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  "*.model",
  "vocab.txt",
  "merges.txt",
)

MODEL_PATTERNS = [
  "*.safetensors*",
  "config.json",
  "generation_config.json",
]


def _upload_directory_to_gcs(local_path: Path, gcs_path: str):
  """Uploads the contents of a local directory to GCS using gsutil.

  Args:
      local_path: The local directory whose contents will be uploaded.
      gcs_path: The destination GCS path (e.g., 'gs://my-bucket/models/').
  """
  if not gcs_path.startswith("gs://"):
    raise ValueError("GCS path must start with gs://")

  logger.info(f"Uploading contents of '{local_path}' to '{gcs_path}'...")
  command = ["gsutil", "-m", "cp", "-r", f"{str(local_path).rstrip('/')}/*", gcs_path]
  try:
    subprocess.run(command, check=True, capture_output=True, text=True)
    logger.info(f"Successfully uploaded assets to {gcs_path}.")
  except subprocess.CalledProcessError as e:
    logger.error(f"Failed to upload {local_path} to {gcs_path}. Error: {e.stderr}")
    raise


def save_hf_model_files_to_gcs(
  repo_id: str,
  gcs_path: str,
  file_type: str,
  temp_dir: str | None = None,
):
  """Downloads model or tokenizer files from Hugging Face and uploads them to GCS.

  This function uses `huggingface_hub.snapshot_download` to fetch specific
  files based on predefined patterns for models and tokenizers. The downloaded
  files are then uploaded to the specified GCS path.

  Args:
      repo_id: The ID of the Hugging Face repository (e.g., 'meta-llama/Llama-3-8B-hf').
      gcs_path: The target GCS path for the files (e.g., 'gs://bucket/models/Llama-3-8B-hf').
      file_type: The type of files to download. Must be one of 'tokenizer',
        'model', or 'all'.
      temp_dir: An optional path to a temporary directory for downloading. If
        None, the system's default temporary directory is used.

  Raises:
      ValueError: If an invalid `file_type` is provided.
  """
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
    )

    logger.info(f"Files for '{repo_id}' downloaded locally to '{snapshot_path}'.")

    _upload_directory_to_gcs(Path(snapshot_path), gcs_path)
