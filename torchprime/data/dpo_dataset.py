"""DPO dataset utilities."""

from __future__ import annotations

from typing import Literal

from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizerBase

from .dataset import load_hf_or_json_dataset

TRUNCATE_OPTION = Literal["right", "left", "drop"]


def _pad(
  ids: list[int],
  labels: list[int],
  max_length: int,
  pad_id: int,
) -> tuple[list[int], list[int], list[int]]:
  """Pad IDs and labels to ``max_length``.

  Args:
    ids: Encoded token IDs.
    labels: Label token IDs.
    max_length: Desired sequence length.
    pad_id: Token ID to use for padding.

  Returns:
    Tuple containing padded ``ids``, ``labels`` and the attention mask.
  """
  ids = ids[:max_length]
  labels = labels[:max_length]
  attn = [1] * len(ids)
  # Pad sequences to ``max_length`` and mask out the padding tokens.
  if len(ids) < max_length:
    ids = ids + [pad_id] * (max_length - len(ids))
    labels = labels + [-100] * (max_length - len(labels))
    attn = attn + [0] * (max_length - len(attn))
  return ids, labels, attn


def _tokenize_pair(
  example: dict,
  tokenizer: PreTrainedTokenizerBase,
  *,
  max_length: int,
  truncation: TRUNCATE_OPTION,
) -> dict | None:
  """Tokenize a preference pair.

  Each example contains a ``prompt`` and two completions: ``chosen`` and
  ``rejected``. The completions are concatenated with the prompt and padded to
  ``max_length``.

  Args:
    example: Raw example with ``prompt``, ``chosen`` and ``rejected`` fields.
    tokenizer: Tokenizer used to encode text.
    max_length: Target length for the encoded sequences.
    truncation: Strategy to handle sequences longer than ``max_length``. If
      ``"drop"`` is specified the pair is skipped.

  Returns:
    A dictionary with encoded tensors or ``None`` if the example was dropped.
  """
  prompt = example.get("prompt", "")
  chosen = example.get("chosen")
  rejected = example.get("rejected")
  if chosen is None or rejected is None:
    return None
  prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

  def build(completion: str):
    """Encode completion and append EOS token if necessary."""
    ids = prompt_ids + tokenizer.encode(completion, add_special_tokens=False)
    # Mask out the prompt portion so that only the completion contributes to the loss.
    labels = [-100] * len(prompt_ids) + tokenizer.encode(
      completion, add_special_tokens=False
    )
    if tokenizer.eos_token_id is not None:
      ids.append(tokenizer.eos_token_id)
      labels.append(tokenizer.eos_token_id)
    if len(ids) > max_length:
      if truncation == "drop":
        # Skip examples that overflow the maximum length.
        return None
      if truncation == "left":
        # Keep the last tokens when truncating from the left.
        ids = ids[-max_length:]
        labels = labels[-max_length:]
      else:
        # Default to truncating from the right.
        ids = ids[:max_length]
        labels = labels[:max_length]
    return ids, labels

  built_c = build(chosen)
  built_r = build(rejected)
  if built_c is None or built_r is None:
    return None

  # Fall back to the EOS token when the tokenizer has no dedicated PAD token.
  pad_id = (
    tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None
    else tokenizer.eos_token_id
  )
  ids_c, labels_c = built_c
  ids_r, labels_r = built_r
  # Pad both completions to a fixed ``block_size``.
  ids_c, labels_c, mask_c = _pad(ids_c, labels_c, max_length, pad_id)
  ids_r, labels_r, mask_r = _pad(ids_r, labels_r, max_length, pad_id)
  return {
    "chosen_input_ids": ids_c,
    "chosen_labels": labels_c,
    "chosen_attention_mask": mask_c,
    "rejected_input_ids": ids_r,
    "rejected_labels": labels_r,
    "rejected_attention_mask": mask_r,
  }


def make_dpo_dataset(
  hf_dataset_name: str | None = None,
  hf_dataset_config_name: str | None = None,
  file_dataset_path: str | None = None,
  split: str = "train",
  cache_dir: str | None = None,
  truncation: TRUNCATE_OPTION = "right",
  *,
  tokenizer: PreTrainedTokenizerBase,
  block_size: int,
) -> Dataset:
  """Create a dataset for Direct Preference Optimization.

  The function supports loading data from the Hugging Face hub or from a local
  JSONL file. Each record must contain ``prompt``, ``chosen`` and ``rejected``
  fields which represent a single preference pair.

  Args:
    hf_dataset_name: Optional name of a dataset on the Hugging Face hub.
    hf_dataset_config_name: Optional dataset configuration name.
    file_dataset_path: Optional path to a local JSONL file.
    split: Dataset split to load when using the hub.
    cache_dir: Directory to cache downloaded data.
    truncation: Strategy used when sequences exceed ``block_size``.
    tokenizer: Tokenizer used to encode the examples.
    block_size: Maximum sequence length after tokenization.

  Returns:
    A :class:`datasets.Dataset` containing processed pairs ready for training.
  """
  data = load_hf_or_json_dataset(
    hf_dataset_name=hf_dataset_name,
    hf_dataset_config_name=hf_dataset_config_name,
    file_dataset_path=file_dataset_path,
    split=split,
    cache_dir=cache_dir,
  )

  records = []
  for ex in data:
    out = _tokenize_pair(
      ex,
      tokenizer,
      max_length=block_size,
      truncation=truncation,
    )
    if out is not None:
      records.append(out)
  return Dataset.from_list(records)
