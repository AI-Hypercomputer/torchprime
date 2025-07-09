"""Manages data loading for the CelebA dataset."""

import logging

import torch
import torchvision
from torchvision.transforms import v2

logger = logging.getLogger(__name__)


def get_splits(seed: int = 42):
    """
    Returns deterministic splits of the CelebA dataset for training and testing.

    This function downloads the CelebA dataset and prepares it for an identity
    classification task.

    
    seed: Random seed for reproducibility.
    """
    torch.manual_seed(seed)

    # Standard transforms for image models
    transforms = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load the official 'train' and 'test' splits for identity classification
    train_ds = torchvision.datasets.CelebA(
        root="/tmp/data",
        split="train",
        target_type="identity",
        transform=transforms,
        download=True,
    )

    test_ds = torchvision.datasets.CelebA(
        root="/tmp/data",
        split="test",
        target_type="identity",
        transform=transforms,
        download=True,
    )

    # The trainer expects a `.classes` attribute. CelebA has 10177 identities.
    train_ds.classes = list(range(10177))
    test_ds.classes = list(range(10177))

    return train_ds, test_ds
