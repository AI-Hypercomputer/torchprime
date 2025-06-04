"""Utilities for preparing supervised fine-tuning datasets."""

from __future__ import annotations

import json
from typing import Literal

import fsspec
from datasets import Dataset, DatasetDict, load_dataset
from transformers.tokenization_utils import PreTrainedTokenizerBase

MASK_OPTION = Literal["none", "last", "all"]
FORMAT_OPTION = Literal["prompt_completion", "chat"]


def _read_json_dataset(path: str, split: str) -> Dataset:
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
  """Download and return a dataset from Hugging Face Hub.

  Args:
    name: Name of the dataset on the hub.
    config: Optional configuration name.
    split: Split to load.
    cache_dir: Directory where the dataset cache should live.

  Returns:
    The loaded ``Dataset`` instance for ``split``.
  """

  data = load_dataset(name, config, split=split, cache_dir=cache_dir)
  assert isinstance(data, Dataset | DatasetDict)
  if isinstance(data, DatasetDict):
    data = data[split]
  return data


def _tokenize_prompt_completion(
  example: dict,
  tokenizer: PreTrainedTokenizerBase,
  *,
  mask_mode: MASK_OPTION,
  max_length: int,
  pad: bool,
) -> dict:
  """Tokenize a prompt/completion record.

  Args:
    example: Sample containing ``prompt`` and ``completion`` fields.
    tokenizer: Tokenizer used for encoding.
    mask_mode: How assistant tokens should be masked.
    max_length: Maximum sequence length.
    pad: Whether to pad or truncate the sequence to ``max_length``.

  Returns:
    Mapping with ``input_ids`` and ``labels`` suitable for training.
  """

  prompt = example["prompt"]
  completion = example["completion"]
  prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
  completion_ids = tokenizer.encode(completion, add_special_tokens=False)
  input_ids = prompt_ids + completion_ids
  labels = input_ids.copy()

  if mask_mode == "all" or mask_mode == "last":
    start = len(prompt_ids)
    labels[start:] = [-100] * len(labels[start:])

  if tokenizer.eos_token_id is not None:
    input_ids.append(tokenizer.eos_token_id)
    labels.append(-100 if mask_mode in {"all", "last"} else tokenizer.eos_token_id)

  if pad:
    if len(input_ids) > max_length:
      input_ids = input_ids[:max_length]
      labels = labels[:max_length]
    else:
      pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
      )
      input_ids = input_ids + [pad_id] * (max_length - len(input_ids))
      labels = labels + [-100] * (max_length - len(labels))

  return {"input_ids": input_ids, "labels": labels}


def _tokenize_chat(
  example: dict,
  tokenizer: PreTrainedTokenizerBase,
  *,
  mask_mode: MASK_OPTION,
  max_length: int,
  pad: bool,
) -> dict:
  """Tokenize a conversation in chat format.

  Args:
    example: Sample with a ``messages`` list.
    tokenizer: Tokenizer used for encoding.
    mask_mode: How assistant messages are masked.
    max_length: Maximum sequence length.
    pad: Whether to pad or truncate sequences to ``max_length``.

  Returns:
    Dictionary with ``input_ids`` and ``labels``.
  """

  messages = example["messages"]
  input_ids: list[int] = []
  labels: list[int] = []
  last_assistant = max(
    (i for i, m in enumerate(messages) if m["role"] == "assistant"), default=None
  )

  for idx, message in enumerate(messages):
    ids = tokenizer.encode(message["content"], add_special_tokens=False)
    input_ids.extend(ids)
    mask = message["role"] == "assistant" and (
      mask_mode == "all" or (mask_mode == "last" and idx == last_assistant)
    )
    labels.extend([-100] * len(ids) if mask else ids)

  if tokenizer.eos_token_id is not None:
    input_ids.append(tokenizer.eos_token_id)
    labels.append(
      -100
      if (mask_mode != "none" and last_assistant == len(messages) - 1)
      else tokenizer.eos_token_id
    )

  if pad:
    if len(input_ids) > max_length:
      input_ids = input_ids[:max_length]
      labels = labels[:max_length]
    else:
      pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
      )
      input_ids = input_ids + [pad_id] * (max_length - len(input_ids))
      labels = labels + [-100] * (max_length - len(labels))

  return {"input_ids": input_ids, "labels": labels}


def _pack_samples(examples: list[dict], max_length: int) -> list[dict]:
  """Concatenate and pack tokenized samples.

  Args:
    examples: Sequence of ``{"input_ids", "labels"}`` dicts.
    max_length: Desired packed length.

  Returns:
    List of packed examples. Tokens that do not fill a full block are dropped.
  """

  packed_inputs: list[int] = []
  packed_labels: list[int] = []
  result = []
  for ex in examples:
    packed_inputs.extend(ex["input_ids"])
    packed_labels.extend(ex["labels"])
    while len(packed_inputs) >= max_length:
      result.append(
        {"input_ids": packed_inputs[:max_length], "labels": packed_labels[:max_length]}
      )
      packed_inputs = packed_inputs[max_length:]
      packed_labels = packed_labels[max_length:]
  return result


def make_sft_dataset(
  tokenizer: PreTrainedTokenizerBase,
  max_length: int,
  *,
  hf_name: str | None = None,
  hf_config: str | None = None,
  split: str = "train",
  cache_dir: str | None = None,
  json_path: str | None = None,
  format: FORMAT_OPTION = "prompt_completion",
  mask_mode: MASK_OPTION = "none",
  pack_samples: bool = False,
) -> Dataset:
  """Create a dataset for supervised fine-tuning.

  Either ``hf_name`` or ``json_path`` must be supplied to specify the data
  source. The data can be in plain prompt/completion form or chat format.

  Args:
    tokenizer: Tokenizer used to encode text.
    max_length: Length of padded or packed sequences.
    hf_name: Optional Hugging Face dataset name.
    hf_config: Optional dataset config name.
    split: Dataset split to load.
    cache_dir: Optional directory for HF dataset cache.
    json_path: Optional path or ``gs://`` URI to a JSONL dataset.
    format: ``"prompt_completion"`` or ``"chat"``.
    mask_mode: ``"none"``, ``"last"`` or ``"all"`` assistant masking.
    pack_samples: Whether to pack multiple samples into fixed-length blocks.

  Returns:
    Dataset of tokenized examples ready for model training.
  """
  if hf_name:
    data = _load_hf_dataset(hf_name, hf_config, split, cache_dir)
  elif json_path:
    data = _read_json_dataset(json_path, split)
  else:
    raise ValueError("Either hf_name or json_path must be provided")

  if format == "prompt_completion":
    data = data.map(
      lambda ex: _tokenize_prompt_completion(
        ex, tokenizer, mask_mode=mask_mode, max_length=max_length, pad=not pack_samples
      )
    )
  else:
    data = data.map(
      lambda ex: _tokenize_chat(
        ex, tokenizer, mask_mode=mask_mode, max_length=max_length, pad=not pack_samples
      )
    )

  if pack_samples:
    samples = [ex for ex in data]
    packed = _pack_samples(samples, max_length)
    data = Dataset.from_list(packed)

  return data
