"""BaseCausalLM Module

This module defines a minimal base class for causal language models using PyTorch.
It includes a standard weight initialization method, a placeholder forward pass,
and methods for saving and loading model checkpoints using the `safetensors` format.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import huggingface_hub
import torch
import torch.distributed as dist
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from omegaconf import DictConfig, OmegaConf

from torchprime.torch_xla_models.model import model_utils

logger = logging.getLogger(__name__)


class BaseCausalLM(nn.Module):
  """Base class for causal language models.

  This class provides a template for building causal language models using PyTorch.
  It includes methods for weight initialization, saving, and loading model checkpoints.
  Subclasses should implement the `forward` method.
  """

  config: DictConfig

  def _init_weights(self, module: nn.Module):
    """Initialize weights for Linear and Embedding layers.

    This method initializes the weights of Linear and Embedding layers
    using a normal distribution with mean 0 and standard deviation specified
    by `self.config.initializer_range`. Biases are initialized to zero.

    Args:
        module: The module whose weights need to be initialized.
    """
    std = self.config.initializer_range
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=std)
      if module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
      module.weight.data.normal_(mean=0.0, std=std)
      if module.padding_idx is not None:
        module.weight.data[module.padding_idx].zero_()

  def forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor | None = None,
    attention_mask: torch.FloatTensor | None = None,
  ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
    """Forward method to be implemented by subclass.

    Args:
        input_ids: Input token IDs of shape (batch_size, sequence_length).
        labels (optional): Target labels for computing the loss.
        attention_mask (optional): Attention mask to avoid performing attention on padding token indices.

    Returns:
        tuple: A tuple containing the model's output logits and, optionally, the loss.

    Raises:
        NotImplementedError: If the method is not implemented in the subclass.
    """
    raise NotImplementedError("Subclasses must implement forward")

  def export(self, save_directory: str):
    """Export model weights and config to a directory in sharded safetensors format.

    This method saves the model's state dictionary and configuration to the specified directory.
    The state dictionary is saved in sharded safetensors format, grouped by layer prefix.
    The configuration is saved as a JSON file.

    Note:
        In distributed training setups, ensure that only the primary process
        (e.g., rank 0) performs the saving operation to avoid conflicts.

    Args:
        save_directory: Directory where the model weights and configuration will be saved.
    """
    os.makedirs(save_directory, exist_ok=True)
    state_dict = {
      k: v.cpu() if str(v.device).startswith("xla") else v
      for k, v in self.state_dict().items()
    }
    model_utils.save_sharded_safetensors_by_layer(state_dict, save_directory)

    with open(os.path.join(save_directory, "config.json"), "w") as f:
      json.dump(OmegaConf.to_container(self.config, resolve=True), f, indent=2)

  def from_pretrained(self, model_path_or_repo: str):
    """Load model weights from local directory or Hugging Face Hub repo.

    This method loads the model's state dictionary from the specified path or repository.
    It supports both local directories and remote repositories hosted on the Hugging Face Hub.

    Note:
        In distributed training setups, ensure that all replicas perform the loading operation
        to synchronize model weights across processes.

    Args:
        model_path_or_repo: Path to the local directory or Hugging Face Hub repository ID.
    """
    with model_utils.local_path_from_gcs(
      model_path_or_repo
    ) as local_model_path_or_repo:
      if os.path.isdir(local_model_path_or_repo):
        model_dir = local_model_path_or_repo
      else:
        model_dir = huggingface_hub.snapshot_download(
          repo_id=local_model_path_or_repo,
          allow_patterns=["*.safetensors*"] + model_utils.HF_MODEL_CONFIG_FILES,
        )

      # Load weights
      state_dict = model_utils.load_safetensors_to_state_dict(model_dir)
      self.load_state_dict(state_dict)

  def _maybe_save_checkpoint(self, config: DictConfig) -> None:
    """Save a sharded checkpoint and optionally convert it to safetensors format.

    This method performs the following steps:
      1. Check if export is enabled via `export_checkpoint_path`.
      2. Create the save directory and flush pending XLA device ops.
      3. Save the HF config files and tokenizer.
      4. Initialize a torch.distributed process group if needed.
      5. Save the model state using torch.distributed.checkpoint (DCP).
      6. If `convert_to_safetensors` is enabled and current process is rank 0,
        reload the checkpoint on CPU and export sharded safetensors + index.
      7. Synchronize all processes using an XLA rendezvous barrier.
    """
    # Step 1: Check export path
    folder_name = getattr(config.task, "export_checkpoint_path", None)
    if folder_name is None:
      logger.info("Skipping model export, no export_checkpoint_path provided.")
      return

    # Step 2: Prepare save directory and flush device ops
    save_dir = Path(config.output_dir) / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    xm.mark_step()
    xm.wait_device_ops()

    # Step 3: Save the HF config files and tokenizer
    if xr.process_index() == 0:
      logger.info("Saving Hugging Face configs and tokenizer to %s", save_dir)
      # Copy to local if in GCS
      with (
        model_utils.local_path_from_gcs(config.model.tokenizer_name) as tokenizer_path,
        model_utils.local_path_from_gcs(config.model.pretrained_model) as model_path,
      ):
        model_utils.copy_hf_config_files(tokenizer_path, save_dir)
        model_utils.save_hf_tokenizer(model_path, save_dir)

    # Step 4: Initialize torch.distributed process group
    if not dist.is_initialized():
      xr.use_spmd()
      dist.init_process_group("gloo", init_method="xla://")

    # Step 5: Save distributed checkpoint
    logger.info("Saving distributed checkpoint to %s", save_dir)
    model_utils.save_distributed_checkpoint(self, save_dir)

    # Step 6: Convert to safetensors on rank-0 if enabled
    if (
      getattr(config.task, "convert_to_safetensors", False) and xr.process_index() == 0
    ):
      logger.info("Converting distributed checkpoint to safetensors")
      model_utils.convert_to_safetensors_on_cpu(self, save_dir)

    # Step 7: Barrier to synchronize all ranks
    if xr.process_count() > 1:
      xm.rendezvous("sft_save")
