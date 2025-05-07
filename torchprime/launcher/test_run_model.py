import torch

from torchprime.launcher import run_model


class SimpleModel(torch.nn.Module):

  def forward(self, x):
    return x + 100

  def get_sample_inputs(self, batch_size):
    return (torch.randn(batch_size, 10), ), {}


def test_run_model():
  m = SimpleModel()
  times = run_model.run_model_xla(m, 5, 3)
  assert len(times) == 3
  times = run_model.run_model_torchax(m, 5, 3, eager=True)
  assert len(times) == 3
  times = run_model.run_model_torchax(m, 5, 3, eager=False)
  assert len(times) == 3