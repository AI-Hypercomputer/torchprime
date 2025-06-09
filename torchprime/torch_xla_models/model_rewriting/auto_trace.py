import functools
from collections.abc import Callable, Iterator
from typing import TypeVar

import torch.nn as nn
import torch_xla.debug.profiler as xp

T = TypeVar("T", bound=nn.Module)


def auto_trace(
  module: T,
  traced_types: tuple[type, ...] = (nn.Linear,),
) -> T:
  """Change the forward pass of the module tree to automatically call `xp.Trace`.

  module: the module tree to add tracing.
  traced_types: module types to trace. By default, `nn.Linear` layers will be
    patched to call `xp.Trace` with their member name as the argument.
  """
  for name, child in module.named_children():
    if isinstance(child, traced_types):
      original_forward = child.forward
      _patch_module_forward(child, original_forward, name)
    elif isinstance(child, nn.Module) and _not_empty(child.children()):
      original_forward = child.forward
      if not isinstance(child, nn.ModuleList) and not isinstance(child, nn.Sequential):
        _patch_module_forward(child, original_forward, name)
      auto_trace(child, traced_types)

  return module


def _not_empty(it: Iterator) -> bool:
  try:
    next(it)
    return True
  except StopIteration:
    return False


def _patch_module_forward(value: nn.Module, original_forward: Callable, name: str):
  @functools.wraps(original_forward)
  def traced_forward(module_self, *args, **kwargs):
    with xp.Trace(name):
      return original_forward(*args, **kwargs)

  value.forward = traced_forward.__get__(value, type(value))  # type: ignore
