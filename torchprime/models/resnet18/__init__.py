"""ResNet-18 model package."""

import torch

from .model import Resnet18ForClassification


def get_model(
  num_classes: int, num_channels: int = 3, seed: int = 42
) -> tuple[torch.nn.Module, str]:
  """
  Factory function to create a Resnet18 model with a default configuration.

  Args:
      num_classes: The number of output classes for the model's classification head.
      num_channels: The number of input channels for the model. Defaults to 3 for RGB images.
      seed: The random seed for model initialization. Defaults to 42.
  Returns:
      A tuple containing the initialized ResNet-18 model and its hash.
  """
  model = Resnet18ForClassification(
    num_classes=num_classes, num_channels=num_channels, seed=seed
  )
  return model, model.model_hash
