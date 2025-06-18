import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from torchprime.sharding.shard_model import (
  ShardedModule,
  shard_torch_xla_model_from_config,
)


class SimpleLinear(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(128, 64)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(64, 128)

  def forward(self, x):
    y = self.relu(self.fc1(x))
    z = self.fc2(y)
    return z


class MockShardedTensor(torch.Tensor):
  """
  This class simulates a sharded tensor.
  """

  def __init__(self, orig):
    super().__init__()
    self.orig = orig


class MockShardedModule(nn.Module):
  """
  This class simulates an activation (output) sharded module.
  """

  def __init__(self, orig):
    super().__init__()
    self.orig = orig

  def forward(self, x):
    return self.orig(x)


def validate_shard_model_from_config_torch_xla_core(
  sharding_config: dict,
  mesh_shape: tuple,
  mesh_axis: tuple,
  weight_sharding: str,
  activation_sharding: str,
):
  xr.use_spmd()
  model_to_shard = SimpleLinear().to(torch_xla.device())

  # Define mesh for test
  num_devices = xr.global_runtime_device_count()
  assert num_devices > 1, "The TPU VM should have more than 1 device for SPMD testing"
  device_ids = np.array(range(num_devices))
  mesh = Mesh(device_ids, mesh_shape, mesh_axis)
  model = shard_torch_xla_model_from_config(model_to_shard, sharding_config, mesh)
  torch_xla.sync()

  # In order to shard activations, corresponding modules are
  # wrapped with ShardedModule.
  assert isinstance(model.fc1, ShardedModule)
  assert isinstance(model.fc2, ShardedModule)
  # Check the sharding of weights.
  state_dict = model.state_dict()
  none_sharded = "{replicated}"
  expected_sharding = {
    "fc1._orig_mod.weight": weight_sharding,
    "fc1._orig_mod.bias": none_sharded,
    "fc2._orig_mod.weight": weight_sharding,
    "fc2._orig_mod.bias": none_sharded,
  }
  seen_count = 0
  for name, param in state_dict.items():
    expectation = expected_sharding.get(name)
    if expectation is None:
      continue
    sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(param)
    assert sharding_spec == expectation
    seen_count += 1
  assert seen_count == len(expected_sharding)
  # Run the model and check the sharding of outputs.
  inputs = torch.randn((32, 128), device=torch_xla.device())
  torch_xla.sync()
  output = model(inputs)
  torch_xla.sync()
  assert isinstance(output, torch.Tensor)
  sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(output)
  assert sharding_spec == activation_sharding
