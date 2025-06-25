"""
The data module contains datasets and efficient dataloading implementations.
"""

from .dataset import make_train_dataset
from .dpo_dataset import make_dpo_dataset
from .sft_dataset import make_sft_dataset

DATASET_BUILDERS = {
  "train": make_train_dataset,
  "sft": make_sft_dataset,
  "dpo": make_dpo_dataset,
}

__all__ = [
  "DATASET_BUILDERS",
  "make_train_dataset",
  "make_sft_dataset",
  "make_dpo_dataset",
]
