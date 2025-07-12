"""Manages data loading for the CelebA dataset."""

import collections
import logging

import torch
import tqdm
from datasets import concatenate_datasets
from torch.utils import data
from torchvision.transforms import v2

from torchprime.data import dataset as hf_dataset

logger = logging.getLogger(__name__)

_CELEBA_HF_DATASET_NAME = "flwrlabs/celeba"


class HuggingFaceCelebA(data.Dataset):
  """A wrapper for the HuggingFace CelebA dataset to make it compatible with the vision trainer."""

  def __init__(self, hf_ds, transforms, num_classes: int):
    self.hf_ds = hf_ds
    self.transforms = transforms
    self.num_classes = num_classes

  def __len__(self):
    return len(self.hf_ds)

  def __getitem__(self, idx):
    item = self.hf_ds[idx]
    image = item["image"]
    # The label is the integer index of the celebrity ID. Since celeb_id is a
    # 1-indexed integer, we can just subtract 1 to get a 0-indexed label.
    label = item["celeb_id"] - 1

    if self.transforms:
      image = self.transforms(image)

    return image, label


def _group_and_split_by_celebrity(hf_ds, seed: int):
  """
  Groups the dataset by celebrity and splits each group into train/test sets.

  This function iterates through the dataset once to group images by celebrity,
  then splits each celebrity's images into an 80/20 train/test set. It also
  efficiently calculates the number of unique celebrities in each split.

  Args:
      hf_ds: The Hugging Face dataset to split.
      seed: The random seed for reproducibility.

  Returns:
      A tuple containing the training dataset, test dataset, number of unique
      celebrities in the training set (which is all of them), and the number
      of unique celebrities in the test set (only those with >= 2 images).
  """
  # Each list will hold the per-celebrity Dataset objects (shards) before they
  # are concatenated into the final train and test sets.
  train_shards = []
  test_shards = []

  logger.info("Grouping dataset by celebrity ID. This may take a moment...")
  celeb_id_to_indices = collections.defaultdict(list)
  for i, celeb_id in enumerate(
    tqdm.tqdm(hf_ds["celeb_id"], desc="Grouping by celebrity")
  ):
    celeb_id_to_indices[celeb_id].append(i)

  logger.info(
    f"Splitting {len(celeb_id_to_indices)} celebrities' images into 80/20 train/test sets."
  )

  num_test_celebs = 0
  for celeb_id in tqdm.tqdm(
    sorted(celeb_id_to_indices.keys()), desc="Splitting data by celebrity"
  ):
    indices = celeb_id_to_indices[celeb_id]
    celeb_ds = hf_ds.select(indices)

    # train_test_split requires at least two samples. If a celebrity has only
    # one image, add it directly to the training set.
    if len(celeb_ds) < 2:
      train_shards.append(celeb_ds)
      continue

    split = celeb_ds.train_test_split(test_size=0.2, seed=seed)
    train_shards.append(split["train"])
    test_shards.append(split["test"])
    num_test_celebs += 1

  return (
    concatenate_datasets(train_shards),
    concatenate_datasets(test_shards),
    len(celeb_id_to_indices),
    num_test_celebs,
  )


def get_splits(seed: int = 42):
  """
  Returns deterministic splits of the CelebA dataset for training and testing.

  This function downloads the CelebA dataset from Hugging Face and prepares it.
  for a multi-class celebrity identification task.

  Args:
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

  # Load the full dataset (all splits).
  hf_ds = hf_dataset.load_hf_or_json_dataset(
    hf_dataset_name=_CELEBA_HF_DATASET_NAME, split="train+valid+test"
  )

  logger.info(f"Loaded a total of {len(hf_ds)} images from the CelebA dataset.")

  (hf_train_ds, hf_test_ds, num_train_celebs, num_test_celebs) = (
    _group_and_split_by_celebrity(hf_ds, seed)
  )

  logger.info(
    f"Created training set with {len(hf_train_ds)} images from {num_train_celebs} unique celebrities."
  )
  # We split by celebrity, so both train and test sets cover the same set of celebrities.
  logger.info(
    f"Created test set with {len(hf_test_ds)} images from {num_test_celebs} unique celebrities."
  )

  hf_train_ds = hf_train_ds.shuffle(seed=seed)
  hf_test_ds = hf_test_ds.shuffle(seed=seed)

  logger.info(
    f"Setting up celebrity identification task with {num_train_celebs} classes."
  )
  train_ds = HuggingFaceCelebA(hf_train_ds, transforms, num_classes=num_train_celebs)
  test_ds = HuggingFaceCelebA(hf_test_ds, transforms, num_classes=num_train_celebs)

  return train_ds, test_ds
