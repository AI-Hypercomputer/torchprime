"""Model architecture for ResNet-18."""

import hashlib
import io
import logging
from typing import Any

import torch
import torch.nn as nn
import torchvision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _hash_model_state_dict(model: nn.Module) -> str:
  """
  Creates a SHA256 hash of a model's state_dict for reproducibility.

  Args:
      model: The model to hash.

  Returns:
      A SHA256 hash.
  """
  # Extract the state dict from the model
  state_dict = model.state_dict()

  # Serialize the state_dict to a byte stream
  buffer = io.BytesIO()
  torch.save(state_dict, buffer)
  buffer.seek(0)

  # Get the byte data of the serialized state dict
  state_dict_bytes = buffer.read()

  # Hash the byte data using SHA256
  return hashlib.sha256(state_dict_bytes).hexdigest()


class Resnet18ForClassification(nn.Module):
  """ResNet-18 model with a classification head."""

  def __init__(self, num_classes: int, num_channels: int = 3, seed: int = 42):
    super().__init__()
    self.num_channels = num_channels
    torch.manual_seed(seed)

    # Load a pretrained ResNet-18 model
    self.model = torchvision.models.resnet18(
      weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    num_ftrs = self.model.fc.in_features
    self.model.fc = nn.Linear(num_ftrs, num_classes)
    self.model.conv1 = nn.Conv2d(
      self.num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    self.model.bn1 = nn.BatchNorm2d(64)

    self.model_hash = _hash_model_state_dict(self.model)
    logger.info("Initialized model with seed %d and hash %s", seed, self.model_hash)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Defines the forward pass of the model."""
    return self.model(x)

  def freeze_backbone(self) -> None:
    """Freeze the backbone of the model, leaving the classification head trainable."""
    for param in self.model.parameters():
      param.requires_grad = False

    # Unfreeze the layers we want to train
    for layer in [self.model.fc, self.model.conv1, self.model.bn1]:
      for param in layer.parameters():
        param.requires_grad = True

    logger.info("Backbone of the model has been frozen.")

  def unfreeze_all(self) -> None:
    """Unfreeze all parameters in the model."""
    for param in self.model.parameters():
      param.requires_grad = True
    logger.info("Backbone of the model has been unfrozen.")

  def get_sample_inputs(
    self, batch_size: int
  ) -> tuple[tuple[torch.Tensor], dict[str, Any]]:
    """Returns sample inputs required to run this model."""
    sample_input = torch.rand(
      batch_size,
      self.num_channels,
      224,
      224,
      dtype=torch.float32,
    )
    return (sample_input,), {}
