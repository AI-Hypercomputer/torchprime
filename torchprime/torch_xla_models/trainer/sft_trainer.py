"""Trainer for supervised fine-tuning (SFT) tasks."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from pathlib import Path

import torch

# save_utils.py
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch_xla.core.xla_model as xm
import torch_xla.experimental.distributed_checkpoint as xc
import torch_xla.runtime as xr
from omegaconf import DictConfig
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

from torchprime.torch_xla_models.model.base_causal_lm import (
  save_sharded_safetensors_by_layer,
)

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
    self._maybe_save_model()
    dt = time.perf_counter() - t0
    logger.info("[SAVING] Finished in %.2f s", dt)

  def _maybe_save_model(self) -> None:
    """Save a sharded checkpoint with torch.distributed.checkpoint.

    Call **once** on all TPU ranks at the end of training.

    • All ranks write a sharded *Distributed Checkpoint* (fast) to
      ``<output_dir>/<export_checkpoint_path>/``
    • Optionally, Rank-0 immediately reloads that checkpoint on CPU and emits
      Hugging-Face-compatible `*.safetensors` shards + index.
    """
    folder_name = getattr(self.config.task, "export_checkpoint_path", None)
    if folder_name is None:
      logger.info("Skipping model export, no export_checkpoint_path provided.")
      return

    save_dir = Path(self.config.output_dir) / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Make sure pending device ops are flushed
    xm.mark_step()
    xm.wait_device_ops()

    # Ensure a torch.distributed PG exists (once per host)
    if not dist.is_initialized():
      xr.use_spmd()

      dist.init_process_group("gloo", init_method="xla://")

    # -------------------------- 1 · fast distributed save -------------
    state_dict = {"model": self.model.state_dict()}

    dist_cp.save(
      state_dict=state_dict,
      storage_writer=FileSystemWriter(
        str(save_dir), thread_count=max(2, min(8, mp.cpu_count()))
      ),
      planner=xc.SPMDSavePlanner(),
    )
    logger.info("DCP checkpoint written to %s", save_dir)

    # -------------------------- 2 · CPU safetensor conversion (rank-0) -
    convert_to_safetensors = getattr(self.config.task, "convert_to_safetensors", False)

    if convert_to_safetensors and xr.process_index() == 0:
      logger.info("Rank-0: reloading checkpoint for safetensors export …")

      # build placeholder dict purely from names (no device copies)
      reload_sd = {
        "model": {
          name: torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
          for name, tensor in state_dict["model"].items()
        }
      }

      dist_cp.load(
        state_dict=reload_sd,
        storage_reader=FileSystemReader(str(save_dir)),
        planner=xc.SPMDLoadPlanner(),
      )
      logger.info("Checkpoint fully materialised on CPU")

      cpu_state = {
        k.replace("._orig_mod", ""): v for k, v in reload_sd["model"].items()
      }
      save_sharded_safetensors_by_layer(cpu_state, str(save_dir))
      logger.info("Safetensors shards + index written to %s", save_dir)

    # -------------------------- 3 · barrier so other ranks wait --------
    if xr.process_count() > 1:
      xm.rendezvous("sft_save")
