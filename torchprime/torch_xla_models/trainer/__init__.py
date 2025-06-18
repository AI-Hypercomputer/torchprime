"""Trainer module for Torch XLA models."""

from .base_trainer import Trainer
from .sft_trainer import SFTTrainer

TRAINERS = {
  "train": Trainer,
  "sft": SFTTrainer,
}

__all__ = [
  "TRAINERS",
  "Trainer",
  "SFTTrainer",
]
