"""Trainer for supervised fine-tuning (SFT) tasks."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from omegaconf import DictConfig, OmegaConf
from torch import nn

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

  def train_loop(self, metrics_logger) -> None:
    """Run the base training loop and export the model.

    Args:
      metrics_logger: Instance used to record metrics during training.
    """
    super().train_loop(metrics_logger)

    self._maybe_save_model()
    # self._save_model_w_xm()
    # self._save_model_2()

  def _maybe_save_model(self) -> None:
    """Save the fine-tuned model, if export_checkpoint_path is provided.

    The model is exported to ``output_dir/<export_checkpoint_path>`` on process 0 and the
    rest wait on a rendezvous to ensure the write completes before exiting.
    """
    folder_name = getattr(self.config.task, "export_checkpoint_path", None)
    if folder_name is None:
      return

    save_dir = Path(self.config.output_dir) / folder_name
    if xr.process_index() == 0:
      logger.info("Saving model to %s", save_dir)
      self.model.export(str(save_dir))
    xm.rendezvous("sft_save")

  def _save_model_w_xm(self, folder_name: str | None) -> None:
    """Save the fine-tuned model with xm

    Stream a checkpoint from each replica to rank<N>.pt, then on rank 0 convert
    it to layer-grouped Safetensors shards + index + config.json.
    """
    save_dir = Path(self.config.output_dir) / "trained_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. stream shard from TPU → host; filename is unique per rank
    shard_pt = save_dir / f"rank{xr.process_index()}.pt"
    xm.save(self.model.state_dict(), str(shard_pt))  # non-blocking, low RAM
    xm.rendezvous("sft_save_stream")  # everyone arrives fast

    # 2. only rank 0 merges & converts to Safetensors
    if xr.process_index() == 0:
      # 2-a  load *one* shard (all identical in FSDP/SPMD)
      state = torch.load(shard_pt, map_location="cpu")  # allocates RAM once

      # 2-b  write layer-grouped Safetensors + index
      save_sharded_safetensors_by_layer(state, str(save_dir))

      # 2-c  dump Hydrated config
      config_path = save_dir / "config.json"

      with open(config_path, "w") as f:
        json.dump(OmegaConf.to_container(self.config, resolve=True), f, indent=2)

      # (optional) remove the temporary pt shard to save disk space
      shard_pt.unlink(missing_ok=True)

    # 3. keep other ranks alive until rank 0 finishes file I/O
    xm.rendezvous("sft_save_done")

  def _save_model_2(self, folder_name: str | None):
    save_dir = Path(self.config.output_dir) / "trained_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    if xr.process_index() == 0:  # <- only rank-0 does I/O
      tmp_pt = save_dir / "pytorch_model.pt"
      print("[0] streaming weights to", tmp_pt)
      xm.save(self.model.state_dict(), str(tmp_pt))  # takes ~1–2 min for 8 B

      print("[0] converting to sharded Safetensors")
      import json
      from collections import defaultdict

      import torch
      from safetensors.torch import save_file

      state = torch.load(tmp_pt, map_location="cpu")  # one big CPU copy

      # ----- regroup by layer prefix (your original logic) -----
      def shard_key(name):
        p = name.split(".")
        if p[:2] == ["model", "layers"] and p[2].isdigit():
          return f"model_layers_{p[2]}"
        if p[0] == "model":
          return "model"
        if p[0] == "lm_head":
          return "lm_head"
        return "other"

      groups = defaultdict(dict)
      for k, v in state.items():
        groups[shard_key(k)][k] = v

      weight_map = {}
      for prefix, tensors in groups.items():
        shard_file = f"{prefix}.safetensors"
        save_file(tensors, save_dir / shard_file)
        weight_map.update({k.replace("._orig_mod", ""): shard_file for k in tensors})

      (save_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}, indent=2)
      )

      # save config
      (save_dir / "config.json").write_text(
        json.dumps(OmegaConf.to_container(self.config, resolve=True), indent=2)
      )

      print("[0] checkpoint complete")

    # everybody waits here, but only for rank-0’s single save
    xm.rendezvous("sft_save_done")
