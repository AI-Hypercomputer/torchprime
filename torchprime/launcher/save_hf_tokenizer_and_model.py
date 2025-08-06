"""Utilities for preparing Hugging Face assets (models and tokenizers) for GCS."""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


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


def save_tokenizer_to_gcs(tokenizer_name: str, gcs_path: str):
  """Downloads a tokenizer from a Hugging Face repo and uploads to GCS."""
  with tempfile.TemporaryDirectory() as tmpdir:
    local_path = Path(tmpdir)
    logger.info(f"Created temporary directory: {local_path}")

    # Re-saving the tokenizer ensures it's in a standardized format, which is a good practice.
    # This will only download tokenizer-related files, not the large model weights.
    logger.info(f"Standardizing tokenizer for '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
      tokenizer_name, token=os.environ.get("HF_TOKEN")
    )
    tokenizer.save_pretrained(str(local_path))

    logger.info(f"Tokenizer for '{tokenizer_name}' downloaded and prepared locally.")

    # Upload the directory contents to the specified GCS path.
    _upload_directory_to_gcs(local_path, gcs_path)


def save_model_to_gcs(model_name: str, gcs_path: str, temp_dir: str | None = None):
  """Downloads a model from a Hugging Face repo and uploads to GCS."""
  with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir:
    # tmpdir is the root for the temporary cache.
    logger.info(f"Created temporary directory: {tmpdir}")

    logger.info(f"Downloading model snapshot for '{model_name}'...")
    # We use the temporary directory as the cache_dir to ensure that the
    # files are downloaded directly into it. This avoids a copy operation
    # from the default Hugging Face cache (~/.cache/huggingface) to /tmp,
    # which can fail if /tmp and /home are on different filesystems and /tmp
    # has limited space. The function returns the path to the actual snapshot directory.
    # We explicitly list the patterns for files we need to ensure we don't download
    # unnecessary files like READMEs or large non-safetensors model weights.
    allow_patterns = [
      "*.safetensors*",
      "config.json",
      "generation_config.json",
      "tokenizer*.json",
      "special_tokens_map.json",
    ]
    snapshot_path = snapshot_download(
      repo_id=model_name,
      cache_dir=str(tmpdir),
      token=os.environ.get("HF_TOKEN"),
      allow_patterns=allow_patterns,
    )

    logger.info(f"Model '{model_name}' downloaded locally to '{snapshot_path}'.")

    # Upload the directory contents to the specified GCS path.
    _upload_directory_to_gcs(Path(snapshot_path), gcs_path)
