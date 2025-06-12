from functools import partial

import torch
import torch.nn as nn
from torch_xla.experimental.assume_pure import assume_pure


class PureModule(nn.Module):
  """PureModule wraps a module whose forward pass does not have side-effects.

  `PureModule`s will only be traced once.
  """

  def __init__(self, module: nn.Module) -> None:
    super().__init__()
    self._module = module
    self._pure_forward = assume_pure(partial(_pure_forward, self._module))

  def forward(self, *args, **kwargs):
    params = dict(self._module.named_parameters())
    buffers = dict(self._module.named_buffers())
    return self._pure_forward(params, buffers, args, kwargs)


def _pure_forward(module, params, buffers, args, kwargs):
  return torch.func.functional_call(module, (params, buffers), args, kwargs)
