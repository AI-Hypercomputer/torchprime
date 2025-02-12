import pytest

from torchprime.torch_xla_models import topology


@pytest.fixture(scope="session", autouse=True)
def enable_spmd():
  import torch_xla.runtime as xr

  xr.use_spmd()


def test_basic():
  import torch_xla.runtime as xr

  # TODO(https://github.com/pytorch/xla/issues/8063): `xla_force_host_platform_device_count` doesn't
  # work on PyTorch/XLA. We must run this on the TPU for now.
  if xr.device_type() != "TPU":
    pytest.skip("This test only works on TPU")

  multi_slice = topology.is_multi_slice()
  num_slices = topology.get_num_slices()
  if multi_slice:
    assert num_slices > 1
  else:
    assert num_slices == 1


def test_is_1d_sharding():
  assert topology.is_1d_sharding((1,))
  assert topology.is_1d_sharding((1, 1, 1, 1))
  assert topology.is_1d_sharding((1, 1, 2, 1))
  assert topology.is_1d_sharding((2,))
  assert topology.is_1d_sharding((2, 1))
  assert not topology.is_1d_sharding((2, 2))
  assert not topology.is_1d_sharding((1, 2, 2, 1))
