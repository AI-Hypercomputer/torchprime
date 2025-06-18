import pytest
import torch
import torch.nn as nn

from torchprime.sharding.shard_model import (
  ShardedModule,
  shard_model,
  shard_model_from_config,
  shard_torchax_model_from_config,
)
from torchprime.sharding.testutils import (
  MockShardedModule,
  MockShardedTensor,
  SimpleLinear,
  validate_shard_model_from_config_torch_xla_core,
)


def test_traverse_weights():
  model = SimpleLinear()
  visited = set()

  def shard_weight(weight, name):
    visited.add(name)
    return MockShardedTensor(weight)

  model = shard_model(model, shard_weight, lambda x, _: x)
  assert visited == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}

  # Check that all weights are sharded.
  sharded = set()
  for name, param in model.named_parameters():
    assert isinstance(param.data, MockShardedTensor)
    sharded.add(name)
  assert sharded == visited


def test_traverse_modules():
  model = SimpleLinear()
  visited = set()

  def shard_activation(module, name):
    visited.add(name)
    return MockShardedModule(module)

  model = shard_model(model, lambda x, _: x, shard_activation)
  # Note that the empty string refers to the whole model.
  assert visited == {"", "fc1", "relu", "fc2"}

  # Check that all modules are sharded.
  sharded = set()
  for name, mod in model.named_modules():
    if isinstance(mod, MockShardedModule):
      sharded.add(name)
  assert len(sharded) == len(visited)


def test_traverse_modules_nested():
  model = nn.Sequential(SimpleLinear(), SimpleLinear())
  visited = set()

  def shard_activation(module, name):
    visited.add(name)
    return MockShardedModule(module)

  model = shard_model(model, lambda x, _: x, shard_activation)
  # Note that the empty string refers to the whole model.
  assert visited == {
    "",
    "0",
    "1",
    "0.fc1",
    "0.relu",
    "0.fc2",
    "1.fc1",
    "1.relu",
    "1.fc2",
  }

  # Check that all modules are sharded.
  sharded = set()
  for name, mod in model.named_modules():
    if isinstance(mod, MockShardedModule):
      sharded.add(name)
  assert len(sharded) == len(visited)


def test_shard_model_from_config_mock():
  model = nn.Sequential(SimpleLinear(), SimpleLinear())
  config = {
    "*.fc1": ["fsdp", None],
    "*.relu": ["fsdp", None],
    "*.fc2": ["fsdp", None],
  }

  num_shard_output_calls = 0

  def shard_output(output, spec):
    nonlocal num_shard_output_calls
    assert spec == ("fsdp", None)
    num_shard_output_calls += 1
    return output

  model = shard_model_from_config(model, config, shard_output, lambda x, _: x)

  # Verify that output mark sharding is called for the right number of times.
  # There should be 6 sharding calls, because there are two `SimpleLinear`,
  # and we annotated 3 modules in each.
  inputs = torch.randn((32, 128))
  output = model(inputs)
  assert output.shape == (32, 128)
  assert num_shard_output_calls == 6


def test_nested_spec_converted_to_tuple():
  model = SimpleLinear()
  config = {"fc1": [["fsdp", "tp"], None]}

  captured_spec = None

  def shard_output(output, spec):
    nonlocal captured_spec
    captured_spec = spec
    return output

  model = shard_model_from_config(model, config, shard_output, lambda x, _: x)
  _ = model(torch.randn(1, 128))
  assert captured_spec == (("fsdp", "tp"), None)


def test_shard_model_from_config_multi_output_mock():
  class Foo(nn.Module):
    def __init__(self) -> None:
      super().__init__()

    def forward(self, x):
      return torch.tensor(100), x

  class Bar(nn.Module):
    def __init__(self) -> None:
      super().__init__()

    def forward(self, x):
      return x, torch.tensor(100)

  class MyMod(nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.foo = Foo()
      self.bar = Bar()

    def forward(self, x):
      a, b = self.foo(x)
      c, d = self.bar((a, b))
      return c, d

  model = MyMod()
  config = {
    "foo[0]": ["fsdp", None],
    "bar[1]": ["fsdp", None],
  }

  num_shard_output_calls = 0

  def shard_output(output, spec):
    nonlocal num_shard_output_calls
    assert spec == ("fsdp", None)
    torch.testing.assert_close(output, torch.tensor(100))
    num_shard_output_calls += 1
    return output

  model = shard_model_from_config(model, config, shard_output, lambda x, _: x)

  # Verify that output mark sharding is called for the right number of times.
  # There should be 2 sharding calls for `foo` and `bar` in total.
  x = torch.tensor(42)
  c, d = model(x)
  torch.testing.assert_close(d, torch.tensor(100))
  a, b = c
  torch.testing.assert_close(a, torch.tensor(100))
  torch.testing.assert_close(b, x)
  assert num_shard_output_calls == 2


def test_shard_model_from_config_torchax():
  # Create 4 CPU devices for SPMD
  with temporary_env({"XLA_FLAGS": "--xla_force_host_platform_device_count=4"}):
    import jax
    import torchax
    import torchax.interop
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh
    from torchax.interop import JittableModule, jax_view

    with torchax.default_env():
      model = SimpleLinear().to("jax")

    config = {
      "fc1": ["fsdp", None],
      "fc1.weight": [None, "fsdp"],
      "fc1.bias": [None],
      "fc2": ["fsdp", None],
      "fc2.weight": [None, "fsdp"],
      "fc2.bias": [None],
    }

    # Define mesh for test
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, ("fsdp",))

    model = shard_torchax_model_from_config(model, config, mesh)

    # In order to shard activations, corresponding modules are
    # wrapped with ShardedModule.
    assert isinstance(model.fc1, ShardedModule)
    assert isinstance(model.fc2, ShardedModule)

    # Check the sharding of weights.
    state_dict = model.state_dict()
    expected_sharding = {
      "fc1._orig_mod.weight": (None, "fsdp"),
      "fc1._orig_mod.bias": (None,),
      "fc2._orig_mod.weight": (None, "fsdp"),
      "fc2._orig_mod.bias": (None,),
    }
    seen_count = 0
    for name, param in state_dict.items():
      param = jax_view(param.data)
      expectation = expected_sharding.get(name)
      if expectation is None:
        continue
      assert param.sharding.spec == expectation
      seen_count += 1
    assert seen_count == len(expected_sharding)

    # Run the model and check the sharding of outputs.
    jit_model = JittableModule(model)
    with torchax.default_env():
      inputs = torch.randn((32, 128), device="jax")
      output = jit_model(inputs)

    assert isinstance(output, torch.Tensor)
    assert jax_view(output).sharding.spec == ("fsdp",)


def test_shard_model_from_config_torch_xla_fsdp():
  import torch_xla.runtime as xr

  if xr.device_type() != "TPU":
    pytest.skip("This test only works on TPU")
  num_devices = xr.global_runtime_device_count()
  validate_shard_model_from_config_torch_xla_core(
    sharding_config={
      "fc1": ["fsdp", None],
      "fc1.weight": [None, "fsdp"],
      "fc1.bias": [None],
      "fc2": ["fsdp", None],
      "fc2.weight": [None, "fsdp"],
      "fc2.bias": [None],
    },
    mesh_shape=(num_devices,),
    mesh_axis=("fsdp",),
    weight_sharding=(
      f"{{devices=[1,{num_devices}]{','.join(str(v) for v in range(num_devices))}}}"
    ),
    activation_sharding=(
      f"{{devices=[{num_devices},1]{','.join(str(v) for v in range(num_devices))}}}"
    ),
  )


def test_shard_model_from_config_torch_xla_cp():
  import torch_xla.runtime as xr

  if xr.device_type() != "TPU":
    pytest.skip("This test only works on TPU")
  context_parallel_size = 2
  num_devices = xr.global_runtime_device_count()
  validate_shard_model_from_config_torch_xla_core(
    sharding_config={
      "fc1": ["fsdp", "context"],
      "fc1.weight": [None, "fsdp"],
      "fc1.bias": [None],
      "fc2": ["fsdp", "context"],
      "fc2.weight": [None, "fsdp"],
      "fc2.bias": [None],
    },
    mesh_shape=(num_devices // context_parallel_size, context_parallel_size),
    mesh_axis=("fsdp", "context"),
    weight_sharding=(
      f"{{devices=[1,{num_devices // context_parallel_size},{context_parallel_size}]{','.join(str(v) for v in range(num_devices)) + ' last_tile_dim_replicate'}}}"
    ),
    activation_sharding=(
      f"{{devices=[{num_devices // context_parallel_size},{context_parallel_size}]{','.join(str(v) for v in range(num_devices))}}}"
    ),
  )


def test_shard_model_from_config_torch_xla_dp():
  import torch_xla.runtime as xr

  if xr.device_type() != "TPU":
    pytest.skip("This test only works on TPU")

  fsdp_size = 2
  num_devices = xr.global_runtime_device_count()
  device_arr = []
  for i in range(num_devices):
    if i % 2 == 0:
      device_arr.append(i)
  for i in range(num_devices):
    if i % 2 != 0:
      device_arr.append(i)

  validate_shard_model_from_config_torch_xla_core(
    sharding_config={
      "fc1": [["data", "fsdp"], None],
      "fc1.weight": [None, "fsdp"],
      "fc1.bias": [None],
      "fc2": [["data", "fsdp"], None],
      "fc2.weight": [None, "fsdp"],
      "fc2.bias": [None],
    },
    mesh_shape=(num_devices // fsdp_size, fsdp_size),
    mesh_axis=("data", "fsdp"),
    weight_sharding=(
      f"{{devices=[1,{fsdp_size},{num_devices // fsdp_size}]{','.join(str(v) for v in device_arr) + ' last_tile_dim_replicate'}}}"
    ),
    activation_sharding=(
      f"{{devices=[{num_devices},1]{','.join(str(v) for v in range(num_devices))}}}"
    ),
  )


def test_shard_model_from_config_torch_xla_mixed():
  import torch_xla.runtime as xr

  if xr.device_type() != "TPU":
    pytest.skip("This test only works on TPU")
  fsdp_size = 2
  cp_size = 2
  num_devices = xr.global_runtime_device_count()
  if num_devices != 8:
    return  # only run the test when the num device is 8

  validate_shard_model_from_config_torch_xla_core(
    sharding_config={
      "fc1": [["data", "fsdp"], "context"],
      "fc1.weight": [None, "fsdp"],
      "fc1.bias": [None],
      "fc2": [["data", "fsdp"], "context"],
      "fc2.weight": [None, "fsdp"],
      "fc2.bias": [None],
    },
    mesh_shape=(num_devices // (fsdp_size * cp_size), fsdp_size, cp_size),
    mesh_axis=("data", "fsdp", "context"),
    # The fsdp is the second dimension of the device mesh, so the device idx that has the second dimension to be the same
    # will hold the same chunk of the data, for example, num_device is 8, for a device mesh (a, b, c), if b is the same,
    # then they hold the same chunk, in this case, 0, 1, 4, 5 hold the same chunk-0, and 2, 3, 6, 7 hold chunk-1
    weight_sharding=(
      f"{{devices=[1,{fsdp_size},{num_devices // fsdp_size}]{','.join(str(v) for v in [0, 1, 4, 5, 2, 3, 6, 7]) + ' last_tile_dim_replicate'}}}"
    ),
    activation_sharding=(
      f"{{devices=[{num_devices // cp_size},{cp_size}]{','.join(str(v) for v in range(num_devices))}}}"
    ),
  )


def temporary_env(env_dict):
  import os
  from contextlib import contextmanager

  @contextmanager
  def _temporary_env(env_dict):
    old_env = {}
    for key, value in env_dict.items():
      old_env[key] = os.environ.get(key)
      os.environ[key] = value
    try:
      yield
    finally:
      for key, value in old_env.items():
        if value is None:
          del os.environ[key]
        else:
          os.environ[key] = value

  return _temporary_env(env_dict)


if __name__ == "__main__":
  test_shard_model_from_config_torch_xla_mixed()
