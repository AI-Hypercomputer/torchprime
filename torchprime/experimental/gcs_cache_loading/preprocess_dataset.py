import logging

from transformers import AutoTokenizer

from torchprime.data.dataset import make_train_dataset

logger = logging.getLogger(__name__)


def main(
  dataset_name: str,
  dataset_config_name: str | None,
  tokenizer_name: str,
  output_path: str,
  block_size: int,
  num_proc: int,
  split: str,
  text_column: str,
) -> None:
  """Main function to preprocess a dataset and save it to a specified location.

  Args:
      dataset_name: Name of the Hugging Face dataset.
      dataset_config_name: Optional configuration name for the dataset.
      tokenizer_name: Name of the Hugging Face tokenizer.
      output_path: Path to save the processed dataset.
      block_size: Sequence length for packing.
      num_proc: Number of processes for mapping.
      split: Dataset split to process.
      text_column: The column containing text data.
  """
  logger.info("Starting dataset preprocessing...")

  logger.info(f"Loading tokenizer: {tokenizer_name}")
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  processed_dataset = make_train_dataset(
    hf_dataset_name=dataset_name,
    hf_dataset_config_name=dataset_config_name,
    split=split,
    tokenizer=tokenizer,
    block_size=block_size,
    text_column=text_column,
    num_proc=num_proc,
    streaming=False,
  )

  logger.info(f"Saving processed dataset to: {output_path}")
  processed_dataset.save_to_disk(output_path)
  logger.info("Preprocessing complete.")