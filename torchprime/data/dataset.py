"""Utilities for preparing datasets for basic training tasks."""

import json

import fsspec
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers.tokenization_utils import PreTrainedTokenizerBase

from torchprime.torch_xla_models.model import model_utils


def _load_preprocessed_dataset(path: str, split: str, cache_dir: str | None) -> Dataset:
  """Loads a `datasets` object from a directory saved with `save_to_disk`.

  Handles both local paths and GCS URIs. If a GCS path is provided, the data is
  first downloaded to a local temporary directory. If the loaded object is a
  `DatasetDict`, the specified `split` is returned.

  Args:
    path: The path to the dataset directory (local or GCS).
    split: The dataset split to load.
    cache_dir: Optional temporary directory for GCS downloads.

  Returns:
    The loaded `datasets.Dataset` object.
  """
  if path.startswith("gs://"):
    with model_utils.local_path_from_gcs(path, temp_dir=cache_dir) as local_path:
      data = load_from_disk(local_path)
  else:
    data = load_from_disk(path)
  if isinstance(data, DatasetDict):
    return data[split]
  return data


def _load_json_dataset(path: str, split: str) -> Dataset:
  """Load a dataset from a JSON Lines file.

  Args:
    path: Local path or ``gs://`` URI to the JSONL file.
    split: Unused but kept for API parity with HuggingFace loaders.

  Returns:
    Dataset containing all records from ``path``.
  """

  if path.startswith("gs://"):
    with fsspec.open(path, "r") as f:
      records = [json.loads(line) for line in f]
    return Dataset.from_list(records)

  data = load_dataset("json", data_files=path, split=split)
  assert isinstance(data, Dataset)
  return data


def _load_hf_dataset(
  name: str,
  config: str | None,
  split: str,
  cache_dir: str | None,
) -> Dataset:
  """Load a dataset from Hugging Face Hub or a GCS path.

  If `name` is a GCS path (starts with 'gs://'), it loads a `datasets` object
  from a directory saved with `save_to_disk`. Otherwise, it downloads and
  returns a dataset from Hugging Face Hub.

  Args:
    name: Name of the dataset on the hub, or a GCS path to a saved dataset.
    config: Optional configuration name for datasets from the Hub.
    split: Split to load.
    cache_dir: Directory where the dataset cache should live.

  Returns:
    The loaded ``Dataset`` instance for ``split``.
  """

  if name.startswith("gs://"):
    with model_utils.local_path_from_gcs(name, temp_dir=cache_dir) as local_path:
      data = load_from_disk(local_path)
  else:
    data = load_dataset(name, config, split=split, cache_dir=cache_dir)
  assert isinstance(data, Dataset | DatasetDict)
  if isinstance(data, DatasetDict):
    data = data[split]
  return data


def _load_raw_dataset(
  path_or_name: str,
  config_name: str | None = None,
  split: str = "train",
  cache_dir: str | None = None,
):
  """Loads a raw dataset from Hugging Face Hub, a GCS path, or a JSONL file.

  This function abstracts the logic for loading datasets from two sources:
  1. Hugging Face Hub or a pre-saved GCS dataset directory.
  2. A JSONL file (local or `gs://`-hosted).

  Args:
    path_or_name: Name of the HF dataset, or a path to a JSONL file or GCS directory.
    config_name: Optional configuration name for the HF dataset.
    split: Dataset split to load (default is "train").
    cache_dir: Optional directory to use for dataset caching (HF only).

  Returns:
    A HuggingFace ``Dataset`` instance.
  """
  if path_or_name.endswith((".json", ".jsonl")):
    data = _load_json_dataset(path_or_name, split)
  else:
    data = _load_hf_dataset(path_or_name, config_name, split, cache_dir)

  assert isinstance(data, Dataset), "Loaded dataset must be a Dataset instance."

  return data


def make_train_dataset(
  dataset_path_or_name: str,
  dataset_config_name: str | None = None,
  is_preprocessed: bool = False,
  split: str = "train",
  cache_dir: str | None = None,
  *,
  tokenizer: PreTrainedTokenizerBase,
  block_size: int,
) -> Dataset:
  """Load and prepare a dataset for causal language model training.

  This function downloads a dataset from the Hugging Face Hub, tokenizes the `text`
  column using the provided tokenizer, and groups the resulting tokens into
  contiguous blocks of fixed length (`block_size`). This block-wise packing is useful
  for efficient language modeling, especially on accelerators like TPUs.

  If `is_preprocessed` is True, the function loads a dataset directly from the
  path specified in `dataset_path_or_name`, skipping tokenization.

  Args:
    dataset_path_or_name: HF dataset name, or a path to a local/GCS dataset.
    dataset_config_name: Optional HF dataset config name. (e.g., "wikitext-103-raw-v1").
    is_preprocessed: If True, load a pre-tokenized dataset directly from the path.
    split: Dataset split to load from HF. (e.g., "train", "validation").
    cache_dir: Optional directory for HF dataset cache.
    tokenizer: A Hugging Face tokenizer used to tokenize the input text.
    block_size: The fixed length of each chunked training example.

  Returns:
    A `Dataset` object with tokenized and block-wise grouped training examples.
  """
  if is_preprocessed:
    data = _load_preprocessed_dataset(dataset_path_or_name, split, cache_dir)
    if "input_ids" not in data.features or "labels" not in data.features:
      raise ValueError(
        "Pre-processed dataset is missing 'input_ids' or 'labels' column, which are required for training."
      )
    return data

  data = _load_raw_dataset(
    path_or_name=dataset_path_or_name,
    config_name=dataset_config_name,
    split=split,
    cache_dir=cache_dir,
  )

  column_names = list(data.features)
  data = data.map(
    lambda samples: tokenizer(samples["text"]),
    batched=True,
    remove_columns=column_names,
  )

  def group_texts(examples):
    """Concatenates tokenized texts and chunks them into blocks of `block_size`.

    Taken from run_clm.py. It's important to group texts evenly to avoid recompilations in TPU.
    """
    from itertools import chain

    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.

    result = {
      k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
      for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

  data = data.map(group_texts, batched=True)
  return data
