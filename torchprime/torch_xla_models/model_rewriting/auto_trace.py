from types import MethodType
from typing import TypeVar

import torch.nn as nn
import torch_xla.debug.profiler as xp

T = TypeVar("T", bound=nn.Module)


def auto_trace(
  module: T,
  traced_types: tuple[type, ...] = (nn.Linear,),
) -> T:
  """Insert `xp.Trace` in the forward pass of the module tree.

  module: the module tree to add tracing.
  traced_types: module types to trace. By default, `nn.Linear` layers will be
    patched to call `xp.Trace` with their member name as the argument.
  """
  for name, child in module.named_children():
    if isinstance(child, traced_types):
      _patch_module_forward(child, name)
    elif isinstance(child, nn.Module):
      auto_trace(child, traced_types)

  return module


def _patch_module_forward(module: nn.Module, name: str):
  def traced_forward(module_self, *args, **kwargs):
    with xp.Trace(name):
      return module_self._auto_trace_forward_original(*args, **kwargs)  # type: ignore

  module._auto_trace_forward_original = module.forward  # type: ignore
  module.forward = MethodType(traced_forward, module)
