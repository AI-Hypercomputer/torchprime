import torch
import torch.nn as nn
from torch_xla.experimental.scan_layers import scan_layers

from torchprime.layers.sequential import HomogeneousSequential, PyTree, splat


class HomogeneousSequentialScan(HomogeneousSequential):
  def forward(self, *input, **broadcasted_inputs: PyTree):
    layers = [BroadcastArguments(m) for m in self.children()]
    if len(input) == 1:
      # Handle single argument case: we don't need to call the module with a tuple.
      input = input[0]
    out, _broadcasted_inputs_back = scan_layers(layers, (input, broadcasted_inputs))
    return out


class BroadcastArguments(torch.nn.Module):
  def __init__(self, mod: nn.Module):
    super().__init__()
    self.mod = mod

  def forward(self, orig_input, broadcasted_inputs):
    out = self.mod(*splat(orig_input), **broadcasted_inputs)
    return (out, broadcasted_inputs)


def compile_one_stack(mod: HomogeneousSequential) -> HomogeneousSequential:
  # Replace base class with our optimized subclass.
  if isinstance(mod, HomogeneousSequentialScan):
    return mod
  new_mod = HomogeneousSequentialScan(*mod.children())
  return new_mod


def compile(mod: nn.Module, sequential_to_scan: str) -> nn.Module:
  seq = mod.get_submodule(sequential_to_scan)
  if not isinstance(seq, HomogeneousSequential):
    raise ValueError(f"compile only supports HomogeneousSequential, got {type(seq)}")
  # Replace the submodule
  mod.set_submodule(sequential_to_scan, compile_one_stack(seq))
  return mod
