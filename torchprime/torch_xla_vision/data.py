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

  def __init__(self, hf_ds, transforms):
    self.hf_ds = hf_ds
    self.transforms = transforms
    # The trainer expects a `.classes` attribute. CelebA has 10177 identities.
    self.classes = list(range(10177))

  def __len__(self):
    return len(self.hf_ds)

  def __getitem__(self, idx):
    item = self.hf_ds[idx]
    image = item["image"]
    # 'celeb_id' is the identity label, but it's 1-indexed, make it 0 index
    label = item["celeb_id"] - 1

    if self.transforms:
      image = self.transforms(image)

    return image, label


def get_splits(seed: int = 42):
  """
  Returns deterministic splits of the CelebA dataset for training and testing.

  This function downloads the CelebA dataset from Hugging Face and prepares it
  for an identity classification task.


  seed: Random seed for reproducibility.
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

  train_ds = HuggingFaceCelebA(hf_train_ds, transforms)
  test_ds = HuggingFaceCelebA(hf_test_ds, transforms)

  return train_ds, test_ds
