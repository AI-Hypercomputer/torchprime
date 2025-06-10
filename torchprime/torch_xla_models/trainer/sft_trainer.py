"""Trainer for supervised fine-tuning (SFT) tasks."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from pathlib import Path

import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch_xla.core.xla_model as xm
import torch_xla.experimental.distributed_checkpoint as xc
import torch_xla.runtime as xr
from omegaconf import DictConfig
from torch import nn
from torch.distributed.checkpoint import FileSystemWriter

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

    t0 = time.perf_counter()
    logger.info("[SAVING] Starting distributed checkpoint …")
    self._maybe_save_model_xla_dist()  # For LLAMA-3-8b: Local VM 60.18s |  xpk 356.57s
    dt = time.perf_counter() - t0
    logger.info("[SAVING] Finished in %.2f s", dt)

  def _maybe_save_model_xla_dist(self) -> None:
    """Save a sharded checkpoint with torch.distributed.checkpoint.

    Each TPU core writes its own shard concurrently, avoiding the gather-to-host
    and single-rank I/O bottleneck of ``xm.save()``.
    """
    folder_name = getattr(self.config.task, "export_checkpoint_path", None)
    if folder_name is None:
      logger.info("Skipping model export, no export_checkpoint_path provided.")
      return

    save_dir = Path(self.config.output_dir) / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Flush pending XLA ops so all tensors are materialised
    xm.mark_step()
    xm.wait_device_ops()

    # Initialise a (CPU) process-group exactly once.
    if not dist.is_initialized():
      xr.use_spmd()
      dist.init_process_group("gloo", init_method="xla://")
      print("Distributed process group initialized during saving.")

    # Build the sharded state_dict you want to checkpoint
    state_dict = {
      "model": self.model.state_dict()
    }  # add "optim": opt.state_dict() if desired

    # Synchronous distributed checkpoint
    dist_cp.save(
      state_dict=state_dict,
      storage_writer=FileSystemWriter(
        str(save_dir), thread_count=max(2, min(8, mp.cpu_count()))
      ),
      planner=xc.SPMDSavePlanner(),  # XLA-aware sharding
    )

    logger.info("Distributed checkpoint (sharded) written to %s", save_dir)
