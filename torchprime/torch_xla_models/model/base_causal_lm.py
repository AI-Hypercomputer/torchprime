"""BaseCausalLM Module

This module defines a minimal base class for causal language models using PyTorch.
It includes a standard weight initialization method, a placeholder forward pass,
and methods for saving and loading model checkpoints using the `safetensors` format.
"""

import json
import os
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def load_safetensors_to_state_dict(model_dir: str) -> dict:
  """Load a model state dict from safetensors, supporting both sharded and single-file formats.

  This function loads model weights from the specified directory. It supports both
  sharded (`model.safetensors.index.json`) and single-file (`model.safetensors`) formats.

  Args:
      model_dir: Path to the directory containing the model files.

  Returns:
      dict: A state dictionary containing the model's parameters.

  Raises:
      FileNotFoundError: If neither the sharded nor single-file safetensors are found.
  """

  state_dict = {}
  index_file = os.path.join(model_dir, "model.safetensors.index.json")
  single_file = os.path.join(model_dir, "model.safetensors")

  if os.path.exists(index_file):
    # Load sharded safetensors
    with open(index_file) as f:
      index = json.load(f)
    weight_map = index["weight_map"]
    for filename in set(weight_map.values()):
      path = os.path.join(model_dir, filename)
      with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118
          state_dict[key] = f.get_tensor(key)
  elif os.path.exists(single_file):
    # Load single safetensor file
    state_dict = load_file(single_file)
  else:
    raise FileNotFoundError(
      f"No safetensors found in {model_dir}. Expected 'model.safetensors' or 'model.safetensors.index.json'."
    )

  return state_dict


def save_sharded_safetensors_by_layer(
  state_dict: dict[str, "torch.Tensor"],
  save_dir: str | os.PathLike,
  *,
  max_workers: int = 24,
  tmp_dir: str | os.PathLike | None = None,
):
  """Save a model state dict to sharded safetensors by layer prefix.

  This function saves the model's state dictionary into separate sharded files,
  grouped by the top-level layer prefix. It also creates an index file
  (`model.safetensors.index.json`) mapping each parameter to its corresponding shard.

  Args:
      state_dict (dict): The model's state dictionary to be saved.
      save_dir (str): Directory where the sharded safetensors and index file will be saved.
      max_workers: Parallel writer threads.  24 saturates v6e-4 dual NVMe; tune as needed.
      tmp_dir: If given, write shards to this *local* directory first, then
        `gsutil -m cp` the results to ``save_dir``.  Handy when `save_dir` is a
        Cloud-Storage mount and you want full NVMe speed.
  """
  save_dir = Path(save_dir)
  if tmp_dir:
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    work_dir = tmp_dir
  else:
    save_dir.mkdir(parents=True, exist_ok=True)
    work_dir = save_dir
  work_dir.mkdir(parents=True, exist_ok=True)

  def _shard_key(param_name: str) -> str:
    """Determine the shard key for a parameter based on its name."""
    parts = param_name.split(".")
    if parts[:2] == ["model", "layers"] and parts[2].isdigit():
      return "_".join(parts[:3])  # model_layers_0, model_layers_1, etc.
    elif parts[0] == "model":
      return "model"
    elif parts[0] == "lm_head":
      return "lm_head"
    else:
      return "other"

  grouped: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
  sizes: dict[str, int] = {}
  for k, v in state_dict.items():
    p = _shard_key(k)
    grouped[p][k] = v
    sizes[p] = sizes.get(p, 0) + v.numel() * v.element_size()

  def _write_one(item: tuple[str, dict[str, "torch.Tensor"]]) -> dict[str, str]:
    prefix, group = item
    fname = f"{prefix}.safetensors"
    save_file(group, str(work_dir / fname))
    # strip FSDP suffix for HF compatibility
    return {k.replace("._orig_mod", ""): fname for k in group}

  # sort largest → smallest so threads finish together
  items = sorted(grouped.items(), key=lambda kv: sizes[kv[0]], reverse=True)

  weight_map: dict[str, str] = {}
  with ThreadPoolExecutor(max_workers=max_workers) as pool:
    for mapping in pool.map(_write_one, items):
      weight_map.update(mapping)

  # ---------- dump index --------------------------------------------
  (work_dir / "model.safetensors.index.json").write_text(
    json.dumps({"weight_map": weight_map}, indent=2)
  )

  # ---------- optional gsutil sync ---------------------------------
  if tmp_dir:
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
      "gsutil",
      "-m",
      "-q",
      "cp",
      "-n",  # don't clobber if file exists
      *(str(p) for p in work_dir.glob("*.safetensors")),
      str(save_dir) + "/",
    ]
    cmd_idx = [
      "gsutil",
      "-q",
      "cp",
      str(work_dir / "model.safetensors.index.json"),
      str(save_dir) + "/",
    ]
    subprocess.check_call(cmd)
    subprocess.check_call(cmd_idx)
    shutil.rmtree(work_dir, ignore_errors=True)


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
    save_sharded_safetensors_by_layer(state_dict, save_directory)

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
    if os.path.isdir(model_path_or_repo):
      model_dir = model_path_or_repo
    else:
      model_dir = snapshot_download(
        repo_id=model_path_or_repo, allow_patterns=["*.safetensors*", "config.json"]
      )

    # Load weights
    state_dict = load_safetensors_to_state_dict(model_dir)
    self.load_state_dict(state_dict)
