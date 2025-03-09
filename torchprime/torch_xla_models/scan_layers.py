import torch
import torch.nn as nn
from torch_xla.experimental.scan_layers import scan_layers

from torchprime.layers.sequential import HomogeneousSequential, Input, PyTree


class HomogeneousSequentialScan(HomogeneousSequential):
  def forward(self, input: Input, **broadcasted_inputs: PyTree) -> Input:
    layers = [BroadcastArguments(m) for m in self.children()]
    out, _broadcasted_inputs_back = scan_layers(layers, (input, broadcasted_inputs))
    return out


class BroadcastArguments(torch.nn.Module):
  def __init__(self, mod: nn.Module):
    super().__init__()
    self.mod = mod

  def forward(self, *input):
    orig_input, broadcasted_inputs = input
    out = self.mod(orig_input, **broadcasted_inputs)
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
