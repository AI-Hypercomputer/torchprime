"""Trainer for supervised fine-tuning (SFT) tasks."""

from __future__ import annotations

import logging
from pathlib import Path

import torch_xla
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

  def train_loop(self, metrics_logger) -> None:
    """Run the base training loop and export the model.

    Args:
      metrics_logger: Instance used to record metrics during training.
    """
    super().train_loop(metrics_logger)
    self._save_model()
    # self._save_model_via_xla()

  def _save_model(self) -> None:
    """Save the fine-tuned model.

    The model is exported to ``output_dir/trained_model`` on process 0 and the
    rest wait on a rendezvous to ensure the write completes before exiting.
    """
    save_dir = Path(self.config.output_dir) / "trained_model"
    if xr.process_index() == 0:
      logger.info("Saving model to %s", save_dir)
      self.model.export(str(save_dir))
    xm.rendezvous("sft_save")

  def _save_model_via_xla(self):
    save_dir = Path(self.config.output_dir) / "trained_model_xla"
    # Each host writes its slice; filenames get a replica suffix automatically
    file_path = save_dir / "pytorch_model.pt"  # *.pt, *.bin, *.safetensors all fine
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%d] streaming checkpoint to %s", xr.process_index(), file_path)
    xm.save(self.model.state_dict(), str(file_path))
    # Consolidate on rank 0 if you want a single file
    xm.rendezvous("sft_save")
    if xr.process_index() == 0:
      logger.info("[0] merging per-replica shards → combined file")
      torch_xla.utils.serialization.merge_replica_shards(str(file_path))
