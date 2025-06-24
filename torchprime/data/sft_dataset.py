"""Utilities for preparing supervised fine-tuning datasets."""

from __future__ import annotations

from typing import Literal

from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizerBase

from .dataset import load_hf_or_json_dataset

COMPUTE_OPTION = Literal["all", "completion", "assistant", "last_assistant"]
FORMAT_OPTION = Literal["prompt_completion", "chat"]
TRUNCATE_OPTION = Literal["right", "left", "drop"]


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
    example: Sample containing ``prompt`` and ``completion`` fields,
      where ``prompt`` is the input and ``completion`` is the target output.
      Also supports ``question`` and ``answer``, or simple ``text`` as aliases.
      EXAMPLE:
        {
          "prompt": "What is the capital of France?",
          "completion": "The capital of France is Paris."
        }
        or
        {
          "question": "What is the capital of France?",
          "answer": "The capital of France is Paris."
        }
        or
        {
          "text": "What is the capital of France? The capital of France is Paris."
        }
    tokenizer: Tokenizer used for encoding.
    compute_loss_on: Which parts of the sample should contribute to the loss.
    max_length: Maximum sequence length.
    truncation: ``"right"`` keeps the start, ``"left"`` keeps the end or
      ``"drop"`` removes the sample if it exceeds ``max_length``.

  Returns:
    Mapping with ``input_ids`` and ``labels`` suitable for training.
  """

  if "prompt" in example and "completion" in example:
    prompt = example.get("prompt", "")
    completion = example.get("completion", "")
  elif "question" in example and "answer" in example:
    prompt = example.get("question", "")
    prompt = f"Question:\n{prompt}\n\n\nAnswer:\n"  # Add format for q-a pair
    completion = example.get("answer", "")
  elif "text" in example:
    prompt = ""
    completion = example["text"]
  elif "completion" in example:
    prompt = ""
    completion = example["completion"]
  else:
    raise ValueError(
      "Invalid input format: must contain 'prompt'/'completion' or 'question'/'answer' or 'text' fields."
    )

  prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
  completion_ids = tokenizer.encode(completion, add_special_tokens=False)
  input_ids = prompt_ids + completion_ids
  labels = input_ids.copy()

  if compute_loss_on != "all":
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

  if tokenizer.eos_token_id is not None:
    # always add EOS token and compute loss on it
    input_ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)

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
    example: Sample with a ``messages`` field, containing a list of messages.
      Each message is a dictionary with ``"role"`` (either "user" or "assistant"),
      and ``"content"`` (the text of the message).
      Example:
        {
          "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
            {"role": "user", "content": "What is the population?"}
            {"role": "assistant", "content": "The population of France is about 67 million."}
          ]
        }
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
    # always add EOS token and compute loss on it
    input_ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)

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
  block_size: int,
  *,
  pack: bool,
) -> dict:
  """Pad and optionally pack tokenized samples.

  This helper is compatible with :meth:`datasets.Dataset.map` when
  ``pack=False``. When packing is enabled the returned batch may contain a
  different number of examples, therefore it should be used outside ``map``.

  Args:
    examples: Batch with ``input_ids`` and ``labels`` columns,
      each columns contains a list token IDs for different examples.
      Batched output from _tokenize_chat / _tokenize_prompt_completion methods.
      EXAMPLE:
        {
          "input_ids": [[3,4,5,6,...], [...], ...],
          "labels": [[-100,4,5,6,...], [...], ...]
        }
    tokenizer: Tokenizer providing padding token information.
    block_size: Target sequence/block length.
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
      ids = ids[:block_size]
      labs = labs[:block_size]
      orig_len = len(ids)
      ids = ids + [pad_id] * (block_size - orig_len)
      labs = labs + [-100] * (block_size - len(labs))
      mask = [1] * orig_len + [0] * (block_size - orig_len)
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
    seq_ids = ids[:block_size]
    seq_labels = labs[:block_size]

    if len(cur_ids) + len(seq_ids) > block_size:
      mask = [1] * len(cur_ids) + [0] * (block_size - len(cur_ids))
      result_ids.append(cur_ids + [pad_id] * (block_size - len(cur_ids)))
      result_labels.append(cur_labels + [-100] * (block_size - len(cur_labels)))
      result_mask.append(mask)
      cur_ids, cur_labels = [], []

    cur_ids.extend(seq_ids)
    cur_labels.extend(seq_labels)

    if len(cur_ids) == block_size:
      result_ids.append(cur_ids)
      result_labels.append(cur_labels)
      result_mask.append([1] * block_size)
      cur_ids, cur_labels = [], []

  if cur_ids:
    mask = [1] * len(cur_ids) + [0] * (block_size - len(cur_ids))
    result_ids.append(cur_ids + [pad_id] * (block_size - len(cur_ids)))
    result_labels.append(cur_labels + [-100] * (block_size - len(cur_labels)))
    result_mask.append(mask)

  return {
    "input_ids": result_ids,
    "labels": result_labels,
    "attention_mask": result_mask,
  }


def make_sft_dataset(
  hf_dataset_name: str | None = None,
  hf_dataset_config_name: str | None = None,
  file_dataset_path: str | None = None,
  split: str = "train",
  cache_dir: str | None = None,
  format: FORMAT_OPTION = "prompt_completion",
  compute_loss_on: COMPUTE_OPTION = "completion",
  pack_samples: bool = False,
  truncation: TRUNCATE_OPTION = "right",
  *,
  tokenizer: PreTrainedTokenizerBase,
  block_size: int,
) -> Dataset:
  """Create a dataset for supervised fine-tuning.

  Either ``hf_dataset_name`` or ``file_dataset_path`` must be supplied to specify the data
  source. The data can be in plain prompt/completion form or chat format.

  Args:
    hf_dataset_name: Optional Hugging Face dataset name. (e.g., "wikitext").
    hf_dataset_config_name: Optional HF dataset config name. (e.g., "wikitext-103-raw-v1").
    file_dataset_path: Optional path or ``gs://`` URI to a JSONL dataset.
    split: Dataset split to load from HF. (e.g., "train", "validation").
    cache_dir: Optional directory for HF dataset cache.
    format: ``"prompt_completion"`` or ``"chat"``.
    compute_loss_on: ``"all"``, ``"completion"``, ``"assistant"`` or
      ``"last_assistant"``.
    pack_samples: Whether to pack multiple samples into fixed-length blocks.
    truncation: Strategy for overlong sequences.
    tokenizer: Tokenizer used to encode text.
    block_size: Length of padded or packed sequences.

  Returns:
    Dataset of tokenized examples ready for model training.
  """
  data = load_hf_or_json_dataset(
    hf_dataset_name=hf_dataset_name,
    hf_dataset_config_name=hf_dataset_config_name,
    file_dataset_path=file_dataset_path,
    split=split,
    cache_dir=cache_dir,
  )

  def _tokenize(batch):
    ids = []
    labels = []
    for i in range(len(batch[list(batch.keys())[0]])):
      ex = {k: batch[k][i] for k in batch}
      if format == "prompt_completion":
        out = _tokenize_prompt_completion(
          ex,
          tokenizer,
          compute_loss_on=compute_loss_on,
          max_length=block_size,
          truncation=truncation,
        )
      else:
        out = _tokenize_chat(
          ex,
          tokenizer,
          compute_loss_on=compute_loss_on,
          max_length=block_size,
          truncation=truncation,
        )
      if out is None:
        ids.append(None)
        labels.append(None)
      else:
        ids.append(out["input_ids"])
        labels.append(out["labels"])
    return {"input_ids": ids, "labels": labels}

  data = data.map(_tokenize, batched=True, remove_columns=data.column_names)
  data = data.filter(lambda x: x["input_ids"] is not None)

  if not pack_samples:
    data = data.map(
      _pad_and_maybe_pack_samples,
      batched=True,
      fn_kwargs={"tokenizer": tokenizer, "block_size": block_size, "pack": False},
    )
    return data

  tokenized = {
    "input_ids": [ex["input_ids"] for ex in data],
    "labels": [ex["labels"] for ex in data],
  }
  packed = _pad_and_maybe_pack_samples(tokenized, tokenizer, block_size, pack=True)
  records = [
    {"input_ids": ids, "labels": labs, "attention_mask": mask}
    for ids, labs, mask in zip(
      packed["input_ids"], packed["labels"], packed["attention_mask"], strict=True
    )
  ]
  return Dataset.from_list(records)
