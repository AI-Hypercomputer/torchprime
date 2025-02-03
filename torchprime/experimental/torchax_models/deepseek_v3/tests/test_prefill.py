import pytest


# TODO(https://github.com/AI-Hypercomputer/torchprime/issues/75): Fix the failure on torch 2.6,
# then enable the test unconditionally.
@pytest.mark.deepseek
def test_single_device_compile():
  from torchprime.experimental.torchax_models.deepseek_v3.prefill_benchmark import (
    single_device_compile,
  )

  single_device_compile()


# TODO(https://github.com/AI-Hypercomputer/torchprime/issues/75): Fix the failure on torch 2.6,
# then enable the test unconditionally.
@pytest.mark.deepseek
def test_single_device_eager():
  from torchprime.experimental.torchax_models.deepseek_v3.prefill_benchmark import (
    single_device_eager,
  )

  single_device_eager()
