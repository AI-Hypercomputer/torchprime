"""Trainer for supervised fine-tuning (SFT) tasks."""

from __future__ import annotations

import logging
from pathlib import Path

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
      logger.info(
        "No pretrained model specified; training from scratch. \n\nIs this what you intended?\n"
      )

    super().__init__(model, config, train_dataset)

  def train_loop(self) -> None:
    """Run the base training loop and export the model.

    Args:
      metrics_logger: Instance used to record metrics during training.
    """
    super().train_loop()
    self._maybe_save_model()

  def _maybe_save_model(self) -> None:
    """Save the fine-tuned model, if export_checkpoint_path is provided.

    The model is exported to ``output_dir/<export_checkpoint_path>`` on process 0 and the
    rest wait on a rendezvous to ensure the write completes before exiting.

    Note:
      current exporting and profiling logic is somewhat incompatible, causing race
      conditions if both are enabled. Need to resolve this in future. See issue #297
    """
    folder_name = getattr(self.config.task, "export_checkpoint_path", None)
    if folder_name is None:
      return

    save_dir = Path(self.config.output_dir) / folder_name
    if xr.process_index() == 0:
      logger.info("Saving model to %s", save_dir)
      self.model.export(str(save_dir))
    if xr.process_count() > 1:
      xm.rendezvous("sft_save")
