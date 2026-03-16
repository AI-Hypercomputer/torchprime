"""Trainer module for Torch XLA models."""

from .base_trainer import Trainer
from .dpo_trainer import DPOTrainer
from .sft_trainer import SFTTrainer

TRAINERS = {
  "train": Trainer,
  "sft": SFTTrainer,
  "dpo": DPOTrainer,
}

__all__ = [
  "TRAINERS",
  "Trainer",
  "SFTTrainer",
  "DPOTrainer",
]
