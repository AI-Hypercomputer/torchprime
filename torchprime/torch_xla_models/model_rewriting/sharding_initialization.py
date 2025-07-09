"""Sharding initialization module for TPU-based training using PyTorch/XLA SPMD APIs.

This file defines logic for setting up device mesh topology, determining minibatch support,
and applying weight/activation sharding annotations to the model. It ensures model layers like
`nn.Linear` are patched for SPMD compatibility and leverages user-provided configuration
(OmegaConf) to control sharding behavior.
"""

import logging

import torch.nn as nn
import torch_xla.distributed.spmd as xs
from omegaconf import DictConfig, OmegaConf

from torchprime.sharding.shard_model import shard_torch_xla_model_from_config
from torchprime.torch_xla_models.topology import get_mesh, is_1d_sharding
from torchprime.utils.parallelism_utils import cp_enabled

logger = logging.getLogger(__name__)


def setup_sharding_and_mesh(
  model: nn.Module, config: DictConfig
) -> tuple[nn.Module, xs.ShardingSpec, bool]:
  """Sets up XLA mesh topology and applies SPMD sharding annotations to the model.

  This function:
    - Initializes the global device mesh based on `ici_mesh` in the config.
    - Determines whether minibatch dataloading can be used (only valid for 1D sharding).
    - Creates an input sharding spec indicating how input tensors should be partitioned.
    - Applies a patch to replace `nn.Linear` with einsum-backed versions to preserve dimension semantics.
    - Annotates model weights and intermediate activations with sharding specs.

  Args:
    model: The model to be sharded.
    config: Configuration object specifying mesh and sharding.

  Returns:
    A tuple containing:
      - The sharded model.
      - The input `ShardingSpec` for dataloader inputs.
      - A boolean indicating whether minibatch sharding is supported.
  """
  mesh = get_mesh(config)
  xs.set_global_mesh(mesh)
  logger.info("Logical mesh shape: %s", mesh.shape())
  logger.info("Logical mesh device assignments: %s", mesh.device_ids)

  # TODO(https://github.com/pytorch/xla/issues/8696): Minibatch only works in 1D sharding.
  minibatch = is_1d_sharding(tuple(config.ici_mesh.values()))
  logger.info("Minibatch dataloading: %s", minibatch)

  # TODO(https://github.com/AI-Hypercomputer/torchprime/issues/66): Test this for multislice
  if cp_enabled(config):
    input_sharding_spec = xs.ShardingSpec(
      mesh, (("data", "fsdp"), "context"), minibatch=minibatch
    )
  else:
    input_sharding_spec = xs.ShardingSpec(
      mesh, (("data", "fsdp"), None), minibatch=minibatch
    )

  # Annotate model weights and activations with sharding constraints to distribute
  # the training across devices following the SPMD paradigm.
  sharding_config = OmegaConf.to_container(config.model.sharding, resolve=True)
  assert isinstance(sharding_config, dict)
  model = shard_torch_xla_model_from_config(model, config=sharding_config)

  return model, input_sharding_spec, minibatch
