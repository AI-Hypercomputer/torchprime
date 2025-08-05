"""Utilities for preparing Hugging Face assets (models and tokenizers) for GCS."""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=os.environ.get("HF_TOKEN"))
    tokenizer.save_pretrained(str(local_path))

    logger.info(f"Tokenizer for '{tokenizer_name}' downloaded and prepared locally.")

    # Upload the directory contents to the specified GCS path.
    _upload_directory_to_gcs(local_path, gcs_path)