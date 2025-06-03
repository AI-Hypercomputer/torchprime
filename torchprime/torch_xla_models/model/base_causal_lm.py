"""BaseCausalLM Module

This module defines a minimal base class for causal language models using PyTorch.
It includes a standard weight initialization method, a placeholder forward pass,
and methods for saving and loading model checkpoints using the `safetensors` format.
"""

import json
import os

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file


def load_sharded_safetensors_to_state_dict(model_dir: str) -> dict:
  """Load a model state dict from sharded safetensors in a given directory."""
  state_dict = {}
  index_file = os.path.join(model_dir, "model.safetensors.index.json")
  with open(index_file) as f:
    index = json.load(f)
  weight_map = index["weight_map"]
  for filename in set(weight_map.values()):
    path = os.path.join(model_dir, filename)
    with safe_open(path, framework="pt", device="cpu") as f:
      for key in f.keys():  # noqa: SIM118
        state_dict[key] = f.get_tensor(key)
  return state_dict


def save_sharded_safetensors_by_layer(state_dict: dict, save_dir: str):
  """Save a model state dict to sharded safetensors by layer prefix."""
  os.makedirs(save_dir, exist_ok=True)
  grouped = {}
  for k, v in state_dict.items():
    prefix = k.split(".")[0]
    grouped.setdefault(prefix, {})[k] = v
  weight_map = {}
  for prefix, group in grouped.items():
    shard_file = f"{prefix}.safetensors"
    shard_path = os.path.join(save_dir, shard_file)
    save_file(group, shard_path)
    weight_map.update({k: shard_file for k in group})
  with open(os.path.join(save_dir, "model.safetensors.index.json"), "w") as f:
    json.dump({"weight_map": weight_map}, f, indent=2)


class BaseCausalLM(nn.Module):
  def _init_weights(self, module):
    """Initialize weights for Linear and Embedding layers."""
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
    """Forward method to be implemented by subclass."""
    raise NotImplementedError("Subclasses must implement forward")

  def export(self, save_directory: str):
    """Export model weights and config to a directory in sharded safetensors format."""
    os.makedirs(save_directory, exist_ok=True)
    state_dict = {
      k: v.cpu() if str(v.device).startswith("xla") else v
      for k, v in self.state_dict().items()
    }
    save_sharded_safetensors_by_layer(state_dict, save_directory)

    with open(os.path.join(save_directory, "config.json"), "w") as f:
      json.dump(OmegaConf.to_container(self.config, resolve=True), f, indent=2)

  def from_pretrained(self, model_path_or_repo: str):
    """Load model weights from local directory or Hugging Face Hub repo."""
    if os.path.isdir(model_path_or_repo):
      model_dir = model_path_or_repo
    else:
      model_dir = snapshot_download(
        repo_id=model_path_or_repo, allow_patterns=["*.safetensors*", "config.json"]
      )

    # Load weights
    state_dict = load_sharded_safetensors_to_state_dict(model_dir)
    self.load_state_dict(state_dict)
