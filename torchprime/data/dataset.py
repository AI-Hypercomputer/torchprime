from datasets import Dataset, DatasetDict, load_dataset
from transformers.tokenization_utils import PreTrainedTokenizerBase


def make_huggingface_dataset(
  name: str,
  config_name: str,
  split: str,
  cache_dir: str,
  tokenizer: PreTrainedTokenizerBase,
  block_size: int,
) -> Dataset:
  # Downloading and loading a dataset from the hub.
  data = load_dataset(
    name,
    config_name,
    cache_dir=cache_dir,
  )
  assert isinstance(data, DatasetDict)
  data = data[split]

  column_names = list(data.features)
  data = data.map(
    lambda samples: tokenizer(samples["text"]),
    batched=True,
    remove_columns=column_names,
  )

  # Taken from run_clm.py. It's important to group texts evenly to avoid recompilations in TPU.
  def group_texts(examples):
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
