"""Activation checkpointing, scan compilation, and optimization barrier injection utilities.

This module defines utilities to apply memory-saving techniques during model training,
specifically for use with PyTorch/XLA and SPMD sharding. It includes logic to:

- Add activation rematerialization (checkpointing) to selected layers.
- Compile repeated modules with `scan` for memory/compute efficiency.
- Enable host tensor offloading during checkpointing.
- Apply backward optimization barriers to mitigate recompute overhead.
"""

import logging
from functools import partial

import torch.nn as nn
import torch_xla.distributed.spmd as xs
from omegaconf import DictConfig
from torch_xla.distributed.fsdp import checkpoint_module
from transformers.trainer_pt_utils import get_module_class_from_name

from torchprime.layers.sequential import HomogeneousSequential
from torchprime.sharding.shard_model import wrap_module
from torchprime.torch_xla_models import offloading, remat_all, scan_layers

logger = logging.getLogger(__name__)


def add_activation_checkpointing_and_scan(
  model: nn.Module, config: DictConfig
) -> nn.Module:
  """Applies activation checkpointing and optionally compiles layers using scan.

  This function enables memory-efficient training via:
    - Rematerialization (activation checkpointing) for specified classes.
    - XLA scan compilation for repeated layers to enable loop fusion and lower memory.
    - Optional host offloading of tensors during rematerialization.

  The logic handles multiple modes of operation depending on whether checkpointing,
  scan, and offloading are enabled/configured.

  Args:
    model: Model to transform.
    config: Config with model.remat settings.

  Returns:
    Transformed model with checkpointing and scan if enabled.

  Raises:
    NotImplementedError: If host offloading is enabled without scan.
    NotImplementedError: If multiple layers are passed for offloading.
    NotImplementedError: If checkpointed layer does not match scanned layer.
  """
  remat_config = config.model.remat
  remat_classes = get_classes_by_names(
    model, remat_config.get("activation_checkpoint_layers", [])
  )
  layers_to_scan = remat_config.get("scan_layers", None)
  offload_tensors = remat_config.get("offload_tensors", [])

  # Checking preconditions and logging.
  if remat_classes:
    logger.info("Enabling activation checkpointing on %s", remat_classes)
  if layers_to_scan:
    logger.info("Compiling module `%s` with scan", layers_to_scan)
  if offload_tensors:
    logger.info("Will offload tensors to host RAM: %s", offload_tensors)
    if layers_to_scan is None:
      raise NotImplementedError("Host offloading requires scan")
    if len(remat_classes) != 1:
      raise NotImplementedError(
        "Host offloading requires checkpointing exactly one layer"
      )

  def maybe_checkpoint(mod: nn.Module, _name: str) -> nn.Module:
    if isinstance(mod, tuple(remat_classes)):
      return checkpoint_module(mod)
    return mod

  if layers_to_scan is None:
    return wrap_module(model, maybe_checkpoint) if remat_classes else model

  if not remat_classes:
    return scan_layers.compile(model, layers_to_scan)

  seq = model.get_submodule(layers_to_scan)
  assert isinstance(seq, HomogeneousSequential)
  if list(remat_classes)[0] != seq.repeated_layer:
    raise NotImplementedError("Checkpointing under scan must target scanned layer.")

  partition_fn = (
    remat_all.remat_all_partition_fn
    if not offload_tensors
    else partial(
      offloading.remat_all_and_offload_these_inputs,
      names_to_offload=offload_tensors,
    )
  )
  return scan_layers.compile(model, layers_to_scan, partition_fn=partition_fn)


def add_optimization_barriers(model: nn.Module, config: DictConfig) -> nn.Module:
  """Applies backward optimization barriers to specified layer types.

  Barriers help mitigate recompute inefficiencies during backpropagation.
  They are applied to user-specified layer classes using SPMD APIs.

  Args:
    model: The model to wrap.
    config: Config with model.remat.optimization_barrier_layers.

  Returns:
    Modified model with optimization barriers.
  """
  remat_config = config.model.remat
  classes = get_classes_by_names(
    model, remat_config.get("optimization_barrier_layers", [])
  )
  if not classes:
    return model

  logger.info("Adding backward optimization barriers to %s", classes)

  def maybe_add_barrier(mod: nn.Module, _name: str) -> nn.Module:
    if isinstance(mod, tuple(classes)):
      xs.apply_backward_optimization_barrier(mod)
    return mod

  return wrap_module(model, maybe_add_barrier)


def get_classes_by_names(
  model: nn.Module, class_names: list[str]
) -> tuple[type[nn.Module], ...]:
  """Helper to resolve string class names to actual model classes.

  Args:
    model: Reference model to resolve context.
    class_names: List of fully-qualified class name strings.

  Returns:
    Tuple of resolved class types.

  Raises:
    ValueError: If a class name cannot be resolved within the model.
  """
  classes_to_checkpoint = set()
  for layer_class in class_names:
    cls = get_module_class_from_name(model, layer_class)
    if cls is None:
      raise ValueError(f"Could not find class {layer_class} in model.")
    classes_to_checkpoint.add(cls)
  return tuple(classes_to_checkpoint)
