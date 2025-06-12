import torch
import torch.nn as nn
import torch_xla

from torchprime.torch_xla_models.model_rewriting.assume_pure import PureModule


def test_nn_linear():
  inputs = torch.randn((4,), device="xla")
  linear = nn.Linear(4, 8)
  linear = linear.to("xla")
  expected_output = linear(inputs)
  torch_xla.sync()
  pure_linear = PureModule(linear)
  actual_output = pure_linear(inputs)
  torch_xla.sync()
  torch.testing.assert_close(actual_output, expected_output)
