import torch
import torch.nn as nn
import torch_xla
from omegaconf import OmegaConf

from torchprime.torch_xla_models.model_rewriting.assume_pure import (
  PureModule,
  mark_pure_modules,
)


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


def test_rewrite():
  # Arrange
  class Foo(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, input):
      return input

  class Bar(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, input):
      return input

  class Model(nn.Module):
    def __init__(self):
      super().__init__()
      self.foo = Foo()
      self.bar = Bar()

    def forward(self, input):
      return self.foo(input) + self.bar(input)

  model = Model()
  config = OmegaConf.create(
    {
      "model": {
        "pure_modules": ["Foo"],
      },
    }
  )

  # Act
  model = mark_pure_modules(model, config)

  # Assert
  assert isinstance(model.foo, PureModule)
  assert not isinstance(model.bar, PureModule)
