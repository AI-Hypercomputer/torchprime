"""Model architecture."""

import hashlib
import io
import logging

import torch
import torch.nn as nn
import torchvision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _hash_model_state_dict(model: nn.Module) -> str:
    # Extract the state dict from the model
    state_dict = model.state_dict()

    # Serialize the state_dict to a byte stream
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)  # Go to the start of the buffer

    # Get the byte data of the serialized state dict
    state_dict_bytes = buffer.read()

    # Hash the byte data using SHA256
    model_hash = hashlib.sha256(state_dict_bytes).hexdigest()

    return model_hash


def get_model(num_classes: int, num_channels: int, seed: int) -> float:
    torch.manual_seed(seed)
    # Load a pretrained ResNet18 model
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.conv1 = nn.Conv2d(
        num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.bn1 = nn.BatchNorm2d(64)
    hash = _hash_model_state_dict(model)
    logger.info("Returning model based on seed %d with hash %s", seed, hash)
    return model, hash


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze the backbone of the model."""
    for name, param in model.named_parameters():
        param.requires_grad = False

    for layer in [model.fc, model.conv1, model.bn1]:
        for param in layer.parameters():
            param.requires_grad = True

    logger.info("Backbone of the model has been frozen.")


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Unfreeze the model."""
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Backbone of the model has been unfrozen.")
    return model
