"""Utilities for preparing supervised fine-tuning datasets."""

from __future__ import annotations

import json
from typing import Literal

import fsspec
from datasets import Dataset, DatasetDict, load_dataset
from transformers.tokenization_utils import PreTrainedTokenizerBase

COMPUTE_OPTION = Literal["all", "completion", "assistant", "last_assistant"]
FORMAT_OPTION = Literal["prompt_completion", "chat"]
TRUNCATE_OPTION = Literal["right", "left", "drop"]


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
  compute_loss_on: COMPUTE_OPTION,
  max_length: int,
  truncation: TRUNCATE_OPTION,
) -> dict | None:
  """Tokenize a prompt/completion record.

  Args:
    example: Sample containing ``prompt`` and ``completion`` fields.
    tokenizer: Tokenizer used for encoding.
    compute_loss_on: Which parts of the sample should contribute to the loss.
    max_length: Maximum sequence length.
    truncation: ``"right"`` keeps the start, ``"left"`` keeps the end or
      ``"drop"`` removes the sample if it exceeds ``max_length``.

  Returns:
    Mapping with ``input_ids`` and ``labels`` suitable for training.
  """

  prompt = example["prompt"]
  completion = example["completion"]
  prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
  completion_ids = tokenizer.encode(completion, add_special_tokens=False)
  input_ids = prompt_ids + completion_ids
  labels = input_ids.copy()

  if compute_loss_on != "all":
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

  if tokenizer.eos_token_id is not None:
    input_ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id if compute_loss_on == "all" else -100)

  if len(input_ids) > max_length:
    if truncation == "drop":
      return None
    if truncation == "left":
      input_ids = input_ids[-max_length:]
      labels = labels[-max_length:]
    else:
      input_ids = input_ids[:max_length]
      labels = labels[:max_length]

  return {"input_ids": input_ids, "labels": labels}


def _tokenize_chat(
  example: dict,
  tokenizer: PreTrainedTokenizerBase,
  *,
  compute_loss_on: COMPUTE_OPTION,
  max_length: int,
  truncation: TRUNCATE_OPTION,
) -> dict | None:
  """Tokenize a conversation in chat format.

  Args:
    example: Sample with a ``messages`` list.
    tokenizer: Tokenizer used for encoding.
    compute_loss_on: Which messages should contribute to the loss.
    max_length: Maximum sequence length.
    truncation: ``"right"`` keeps the start, ``"left"`` keeps the end and
      ``"drop"`` removes the sample if it exceeds ``max_length``.

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
    msg_text = tokenizer.apply_chat_template([message], tokenize=False).strip()
    msg_tokens = msg_text.split()
    msg_ids = [tokenizer.convert_tokens_to_ids(t) for t in msg_tokens]
    input_ids.extend(msg_ids)

    if compute_loss_on == "all":
      mask = False
    elif compute_loss_on == "assistant":
      mask = message["role"] != "assistant"
    elif compute_loss_on == "last_assistant":
      mask = not (message["role"] == "assistant" and idx == last_assistant)
    else:  # completion
      mask = idx != len(messages) - 1

    labels.extend([-100] * len(msg_ids) if mask else msg_ids)

  if tokenizer.eos_token_id is not None:
    input_ids.append(tokenizer.eos_token_id)
    mask_last = False
    if compute_loss_on == "assistant":
      mask_last = messages[-1]["role"] != "assistant"
    elif compute_loss_on == "last_assistant":
      mask_last = not (
        messages[-1]["role"] == "assistant" and last_assistant == len(messages) - 1
      )
    elif compute_loss_on == "completion":
      mask_last = False
    labels.append(tokenizer.eos_token_id if not mask_last else -100)

  if len(input_ids) > max_length:
    if truncation == "drop":
      return None
    if truncation == "left":
      input_ids = input_ids[-max_length:]
      labels = labels[-max_length:]
    else:
      input_ids = input_ids[:max_length]
      labels = labels[:max_length]

  return {"input_ids": input_ids, "labels": labels}


def _pad_and_maybe_pack_samples(
  examples: dict,
  tokenizer: PreTrainedTokenizerBase,
  max_length: int,
  *,
  pack: bool,
) -> dict:
  """Pad and optionally pack tokenized samples.

  This helper is compatible with :meth:`datasets.Dataset.map` when
  ``pack=False``. When packing is enabled the returned batch may contain a
  different number of examples, therefore it should be used outside ``map``.

  Args:
    examples: Batch with ``input_ids`` and ``labels`` columns.
    tokenizer: Tokenizer providing padding token information.
    max_length: Target sequence length.
    pack: If ``True`` pack multiple samples together; otherwise pad each sample
      individually.

  Returns:
    A dictionary with ``input_ids``, ``labels`` and ``attention_mask`` lists.
  """

  pad_id = (
    tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None
    else tokenizer.eos_token_id
  )

  ids_list = examples["input_ids"]
  labels_list = examples["labels"]

  if not pack:
    out_ids = []
    out_labels = []
    out_mask = []
    for ids, labs in zip(ids_list, labels_list, strict=True):
      ids = ids[:max_length]
      labs = labs[:max_length]
      orig_len = len(ids)
      ids = ids + [pad_id] * (max_length - orig_len)
      labs = labs + [-100] * (max_length - len(labs))
      mask = [1] * orig_len + [0] * (max_length - orig_len)
      out_ids.append(ids)
      out_labels.append(labs)
      out_mask.append(mask)
    return {"input_ids": out_ids, "labels": out_labels, "attention_mask": out_mask}

  # Packing case handled sequentially, potentially returning fewer sequences.
  result_ids = []
  result_labels = []
  result_mask = []

  cur_ids: list[int] = []
  cur_labels: list[int] = []
  for ids, labs in zip(ids_list, labels_list, strict=True):
    seq_ids = ids[:max_length]
    seq_labels = labs[:max_length]

    if len(cur_ids) + len(seq_ids) > max_length:
      mask = [1] * len(cur_ids) + [0] * (max_length - len(cur_ids))
      result_ids.append(cur_ids + [pad_id] * (max_length - len(cur_ids)))
      result_labels.append(cur_labels + [-100] * (max_length - len(cur_labels)))
      result_mask.append(mask)
      cur_ids, cur_labels = [], []

    cur_ids.extend(seq_ids)
    cur_labels.extend(seq_labels)

    if len(cur_ids) == max_length:
      result_ids.append(cur_ids)
      result_labels.append(cur_labels)
      result_mask.append([1] * max_length)
      cur_ids, cur_labels = [], []

  if cur_ids:
    mask = [1] * len(cur_ids) + [0] * (max_length - len(cur_ids))
    result_ids.append(cur_ids + [pad_id] * (max_length - len(cur_ids)))
    result_labels.append(cur_labels + [-100] * (max_length - len(cur_labels)))
    result_mask.append(mask)

  return {
    "input_ids": result_ids,
    "labels": result_labels,
    "attention_mask": result_mask,
  }


def make_sft_dataset(
  tokenizer: PreTrainedTokenizerBase,
  max_length: int,
  *,
  name: str | None = None,
  config_name: str | None = None,
  data_files: str | None = None,
  split: str = "train",
  cache_dir: str | None = None,
  format: FORMAT_OPTION = "prompt_completion",
  compute_loss_on: COMPUTE_OPTION = "completion",
  pack_samples: bool = False,
  truncation: TRUNCATE_OPTION = "right",
) -> Dataset:
  """Create a dataset for supervised fine-tuning.

  Either ``name`` or ``data_files`` must be supplied to specify the data
  source. The data can be in plain prompt/completion form or chat format.

  Args:
    tokenizer: Tokenizer used to encode text.
    max_length: Length of padded or packed sequences.
    name: Optional Hugging Face dataset name.
    config_name: Optional dataset config name.
    data_files: Optional path or ``gs://`` URI to a JSONL dataset.
    split: Dataset split to load.
    cache_dir: Optional directory for HF dataset cache.
    format: ``"prompt_completion"`` or ``"chat"``.
    compute_loss_on: ``"all"``, ``"completion"``, ``"assistant"`` or
      ``"last_assistant"``.
    pack_samples: Whether to pack multiple samples into fixed-length blocks.
    truncation: Strategy for overlong sequences.

  Returns:
    Dataset of tokenized examples ready for model training.
  """
  if name:
    data = _load_hf_dataset(name, config_name, split, cache_dir)
  elif data_files:
    data = _read_json_dataset(data_files, split)
  else:
    raise ValueError("Either name or data_files must be provided")

  def _tok(batch):
    ids = []
    labels = []
    for i in range(len(batch[list(batch.keys())[0]])):
      ex = {k: batch[k][i] for k in batch}
      if format == "prompt_completion":
        out = _tokenize_prompt_completion(
          ex,
          tokenizer,
          compute_loss_on=compute_loss_on,
          max_length=max_length,
          truncation=truncation,
        )
      else:
        out = _tokenize_chat(
          ex,
          tokenizer,
          compute_loss_on=compute_loss_on,
          max_length=max_length,
          truncation=truncation,
        )
      if out is None:
        ids.append(None)
        labels.append(None)
      else:
        ids.append(out["input_ids"])
        labels.append(out["labels"])
    return {"input_ids": ids, "labels": labels}

  data = data.map(_tok, batched=True, remove_columns=data.column_names)
  data = data.filter(lambda x: x["input_ids"] is not None)

  if not pack_samples:
    data = data.map(
      _pad_and_maybe_pack_samples,
      batched=True,
      fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "pack": False},
    )
    return data

  tokenized = {
    "input_ids": [ex["input_ids"] for ex in data],
    "labels": [ex["labels"] for ex in data],
  }
  packed = _pad_and_maybe_pack_samples(tokenized, tokenizer, max_length, pack=True)
  records = [
    {"input_ids": ids, "labels": labs, "attention_mask": mask}
    for ids, labs, mask in zip(
      packed["input_ids"], packed["labels"], packed["attention_mask"], strict=True
    )
  ]
  return Dataset.from_list(records)
