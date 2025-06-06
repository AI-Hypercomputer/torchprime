"""Utility function(s) for model initialization."""

import importlib
import logging
import re
import sys
from contextlib import contextmanager

import torch


def initialize_model_class(model_config):
  """Import and initialize model_class specified by the config."""
  full_model_class_string = model_config.model_class
  module_name, model_class_name = full_model_class_string.rsplit(".", 1)

  for candidate_module_name in [f"model.{module_name}", module_name]:
    # use full import path to avoid issues with relative imports
    full_module_name = f"torchprime.torch_xla_models.{candidate_module_name}"
    try:
      module = importlib.import_module(full_module_name)
      break
    except ModuleNotFoundError:
      module = None

  if module is None:
    print(f"Error: Failed to import module '{module_name}' or 'model.{module_name}'")
    sys.exit(1)

  if not hasattr(module, model_class_name):
    print(f"Error: Class '{model_class_name}' not found in module '{module.__name__}'")
    sys.exit(1)

  model_class = getattr(module, model_class_name)
  return model_class(model_config)


@contextmanager
def set_default_dtype(dtype: torch.dtype):
  """Temporarily sets the default torch dtype within a context.

  This context manager sets the PyTorch default floating point dtype
  (e.g., `torch.bfloat16`) for the duration of the context
  and restores the original dtype afterward.

  Example:
      ```python
      with set_default_dtype(torch.bfloat16):
          model = MyModel()  # initialized with bfloat16 weights
      ```

  Args:
      dtype: The dtype to set as default within the context.
  """
  previous_dtype = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    yield
  finally:
    torch.set_default_dtype(previous_dtype)


def extract_model_size_from_model_name(model_name: str) -> int | float:
  """Extract the model size in billions from a model name string.

  Args:
      model_name (str): The model name string, e.g., "llama-3-8b.yaml".

  Returns:
      Union[int, float]: The model size in billions, or -1 if not found.
  """
  match = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())
  if match:
    size_str = match.group(1)
    try:
      size = float(size_str)
      return int(size) if size.is_integer() else size
    except ValueError:
      return -1
  return -1


def log_parameter_breakdown(model: torch.nn.Module, logger: logging.Logger) -> None:
  """Logs the number of parameters in different components of the model.

  Args:
      model: The PyTorch model.
      logger: A logger instance to write the output to.
  """
  total_params = sum(p.numel() for p in model.parameters())
  logger.info("Model total size: {} parameters".format(f"{total_params:,}"))

  param_groups = {
    "mlp": 0,
    "attention": 0,
    "embedding": 0,
    "lm_head": 0,
    "norm": 0,
    "other": 0,
  }

  for name, param in model.named_parameters():
    if "mlp" in name:
      param_groups["mlp"] += param.numel()
    elif "self_attn" in name or "attention" in name:
      param_groups["attention"] += param.numel()
    elif "embed" in name:
      param_groups["embedding"] += param.numel()
    elif "lm_head" in name:
      param_groups["lm_head"] += param.numel()
    elif "norm" in name or "layernorm" in name:
      param_groups["norm"] += param.numel()
    else:
      param_groups["other"] += param.numel()

  for k, v in param_groups.items():
    percentage = (v / total_params) * 100
    logger.info("  {:10s}: {} params ({:.2f}%)".format(k, f"{v:,}", percentage))
