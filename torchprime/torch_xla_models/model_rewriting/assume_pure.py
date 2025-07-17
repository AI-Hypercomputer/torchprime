import logging

import torch.nn as nn
import torch_xla.version
from omegaconf import DictConfig
from torch_xla.experimental.assume_pure import PureModule

from torchprime.sharding.shard_model import wrap_module
from torchprime.torch_xla_models.model_rewriting.rematerialization_utils import (
  get_classes_by_names,
)

logger = logging.getLogger(__name__)


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
  if pure_module_config:
    torch_xla_version = torch_xla.version.__version__
    if torch_xla_version.startswith("2.7"):
      logger.warning("pure_modules is not supported for PyTorch/XLA 2.7.x")
      return model
    if torch_xla_version == "2.8.0":
      logger.warning("pure_modules is not supported for PyTorch/XLA 2.8.0")
      return model

  pure_module_classes = get_classes_by_names(model, pure_module_config)

  def transform(mod: nn.Module, _: str):
    if isinstance(mod, pure_module_classes):
      return PureModule(mod)
    return mod

  return wrap_module(model, transform)
