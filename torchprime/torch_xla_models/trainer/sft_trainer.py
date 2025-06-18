"""Trainer for supervised fine-tuning (SFT) tasks."""

from __future__ import annotations

import logging
import time

import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from omegaconf import DictConfig
from torch import nn

from .base_trainer import Trainer

logger = logging.getLogger(__name__)


class SFTTrainer(Trainer):
  """Trainer with pretrained weight loading and saving support."""

  def __init__(
    self,
    model: nn.Module,
    config: DictConfig,
    train_dataset,
  ) -> None:
    """Initialize trainer and optionally load pretrained weights.

    Args:
      model: Model instance to train.
      config: Hydra configuration object.
      train_dataset: Dataset used for training.
    """

    self.pretrained_model = getattr(config.model, "pretrained_model", None)

    if self.pretrained_model:
      if xr.process_index() == 0:
        logger.info("Loading model weights from %s", self.pretrained_model)
      model.from_pretrained(self.pretrained_model)
      xm.mark_step()
    else:
      logger.warning(
        "No pretrained model specified; training from scratch. \n\nIs this what you intended?\n"
      )

    super().__init__(model, config, train_dataset)

  def train_loop(self) -> None:
    """Run the base training loop and export the model.

    Args:
      metrics_logger: Instance used to record metrics during training.
    """
    super().train_loop()

    t0 = time.perf_counter()
    logger.info("[SAVING] Starting distributed checkpoint …")
    self.model._maybe_save_checkpoint(self.config)
    dt = time.perf_counter() - t0
    logger.info("[SAVING] Finished in %.2f s", dt)
