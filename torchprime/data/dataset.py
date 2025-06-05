"""Utilities for preparing datasets for basic training tasks."""

from datasets import Dataset, DatasetDict, load_dataset
from transformers.tokenization_utils import PreTrainedTokenizerBase


def make_train_dataset(
  name: str,
  config_name: str,
  split: str,
  cache_dir: str,
  tokenizer: PreTrainedTokenizerBase,
  block_size: int,
) -> Dataset:
  """Loads and tokenizes a dataset, then chunks it into fixed-size blocks for training.

  This function downloads a dataset from the Hugging Face Hub, tokenizes the `text`
  column using the provided tokenizer, and groups the resulting tokens into
  contiguous blocks of fixed length (`block_size`). This block-wise packing is useful
  for efficient language modeling, especially on accelerators like TPUs.

  Args:
    name: Name of the dataset (e.g., "wikitext").
    config_name: Specific configuration name of the dataset (e.g., "wikitext-103-raw-v1").
    split: Which split to use from the dataset (e.g., "train", "validation").
    cache_dir: Directory to cache the downloaded dataset.
    tokenizer: A Hugging Face tokenizer used to tokenize the input text.
    block_size: The fixed length of each chunked training example.

  Returns:
    A `Dataset` object containing tokenized and block-wise grouped training examples,
    each with keys `"input_ids"` and `"labels"`.
  """
  data = load_dataset(name, config_name, cache_dir=cache_dir)
  assert isinstance(data, DatasetDict)
  data = data[split]

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
