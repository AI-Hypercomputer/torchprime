import pytest
import torch_xla.runtime as xr


def skip_unless_tpu():
  # TODO(https://github.com/pytorch/xla/issues/8063): `xla_force_host_platform_device_count` doesn't
  # work on PyTorch/XLA. We must run this on the TPU for now.
  return pytest.mark.skipif(
    xr.device_type() != "TPU", reason="Test only works works on TPU"
  )
