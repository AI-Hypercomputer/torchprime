"""Manages data loading for the CelebA dataset."""

import logging

import torch
from torch.utils import data
from torchvision.transforms import v2

from torchprime.data import dataset as hf_dataset

logger = logging.getLogger(__name__)

_CELEBA_HF_DATASET_NAME = "flwrlabs/celeba"


class HuggingFaceCelebA(data.Dataset):
  """A wrapper for the HuggingFace CelebA dataset to make it compatible with the vision trainer."""

  def __init__(self, hf_ds, transforms, label_attribute: str = "Bags_Under_Eyes"):
    self.hf_ds = hf_ds
    self.transforms = transforms
    self.label_attribute = label_attribute
    self.classes = [f"No_{label_attribute}", label_attribute]

  def __len__(self):
    return len(self.hf_ds)

  def __getitem__(self, idx):
    item = self.hf_ds[idx]
    image = item["image"]
    label = 1 if item[self.label_attribute] else 0

    if self.transforms:
      image = self.transforms(image)

    return image, label


def get_splits(seed: int = 42, label_attribute: str = "Bags_Under_Eyes"):
  """
  Returns deterministic splits of the CelebA dataset for training and testing.

  This function downloads the CelebA dataset from Hugging Face and prepares it
  for a binary attribute classification task.


  Args:
      seed: Random seed for reproducibility.
      label_attribute: The binary attribute from CelebA to use as the label.
  """
  torch.manual_seed(seed)

  # Standard transforms for image models
  transforms = v2.Compose(
    [
      v2.Resize((224, 224)),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
  )

  # Load the official 'train' and 'test' splits for identity classification
  # from Hugging Face.
  hf_train_ds = hf_dataset.load_hf_or_json_dataset(
    hf_dataset_name=_CELEBA_HF_DATASET_NAME,
    split="train",
  )
  hf_test_ds = hf_dataset.load_hf_or_json_dataset(
    hf_dataset_name=_CELEBA_HF_DATASET_NAME,
    split="test",
  )

  train_ds = HuggingFaceCelebA(hf_train_ds, transforms, label_attribute=label_attribute)
  test_ds = HuggingFaceCelebA(hf_test_ds, transforms, label_attribute=label_attribute)

  return train_ds, test_ds
