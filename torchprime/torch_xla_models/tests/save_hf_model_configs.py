"""Save Hugging Face model configs locally for unit tests."""

import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

# Defines the models whose configs are needed for unit tests.
MODELS_TO_DOWNLOAD = {
  "meta-llama/Meta-Llama-3-8B": "meta-llama-3-8b",
  "meta-llama/Meta-Llama-3.1-405B": "meta-llama-3.1-405b",
  "meta-llama/Llama-4-Scout-17B-16E": "llama-4-scout-17b-16e",
  "mistralai/Mixtral-8x7B-v0.1": "mixtral-8x7b-v0.1",
}

# The root directory for our local test assets.
HF_MODEL_CONFIG_DIR = Path(__file__).parent / "hf_model_config"


def main():
  print(f"Saving model configs to: {HF_MODEL_CONFIG_DIR.resolve()}")

  if not os.environ.get("HF_TOKEN"):
    raise RuntimeError(
      "HF_TOKEN environment variable is not set. "
      "Please set it to download model configs from gated repositories."
    )

  for repo_id, local_dir_name in MODELS_TO_DOWNLOAD.items():
    target_dir = HF_MODEL_CONFIG_DIR / local_dir_name
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading model config for '{repo_id}'...")

    try:
      downloaded_path = hf_hub_download(repo_id=repo_id, filename="config.json")
      shutil.copy2(downloaded_path, target_dir / "config.json")
      print(f"  -> Successfully saved config.json to {target_dir}")
    except Exception as e:
      print(f"  -> FAILED to download config.json for {repo_id}: {e}")


if __name__ == "__main__":
  main()
