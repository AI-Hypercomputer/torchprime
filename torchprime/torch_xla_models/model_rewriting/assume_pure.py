from functools import partial

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_xla.distributed.spmd.xla_sharding import (
  EinsumLinear,
)
from torch_xla.experimental.assume_pure import assume_pure

from torchprime.sharding.shard_model import wrap_module
from torchprime.torch_xla_models.model_rewriting.rematerialization_utils import (
  get_classes_by_names,
)


def mark_pure_modules(model: nn.Module, config: DictConfig) -> nn.Module:
  """Wrap the requested modules in the module tree with `PureModule`.

  There are a few advantages of wrapping a module whose forward pass you know is
  free of side-effects and whose behavior only depends on inputs in a `PureModule`:

  - `PureModule`s will only be traced once.
  - Framework profile scopes added via `xp.Trace` will show up in both the forward
    and the backward pass.

  Args:
    model: Model to transform.
    config: Config with model.pure_modules settings.

  Returns:
    Transformed model.
  """
  pure_module_config = config.model.pure_modules
  pure_module_classes = get_classes_by_names(model, pure_module_config)

  def transform(mod: nn.Module, _: str):
    if isinstance(mod, pure_module_classes):
      return PureModule(mod)
    return mod

  return wrap_module(model, transform)


class PureModule(nn.Module):
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


def replace_nn_linear_with_einsum(module: torch.nn.Module, config: DictConfig):
  """Recursively replace `nn.Linear` layers with `EinsumLinear` in the module.

  Without this patch, an `nn.Linear` module in PyTorch/XLA will lower to reshapes
  and transposes instead of einsum, thus compromising sharding propagation.

  If a `nn.Linear` layer is a (transitive) child of the module class specified in
  `config.model.pure_modules`, or if its type is exactly one of
  `config.model.pure_modules`, then that layer will not be replaced. It is expected that
  it will be traced with torchax via `mark_pure_modules`, which does not have the
  dimension squashing problem.

  TODO: This method can be removed in favor of `xs.apply_xla_patch_to_nn_linear`
  once we teach torchax to lower `torch.ops.xla.einsum_linear_forward`.
  """
  pure_module_config = config.model.pure_modules
  pure_module_classes = get_classes_by_names(module, pure_module_config)

  def recursive_transform(module: torch.nn.Module):
    if isinstance(module, pure_module_classes):
      return
    for name, child in module.named_children():
      if isinstance(child, torch.nn.Linear) and not isinstance(child, EinsumLinear):
        einsum_linear = EinsumLinear(
          child.in_features, child.out_features, bias=child.bias is not None
        )
        einsum_linear.load_state_dict(child.state_dict(), strict=True, assign=True)
        setattr(module, name, einsum_linear)
      elif isinstance(child, nn.Module):
        recursive_transform(child)

  recursive_transform(module)

  return module
