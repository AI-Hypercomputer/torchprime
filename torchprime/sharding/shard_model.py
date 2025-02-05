from collections.abc import Callable

import torch.nn

ShardWeightFn = Callable[[torch.Tensor, str], torch.Tensor]
"""
ShardWeightFn optionally transforms a weight tensor based on its name.

Args:

  weight (torch.Tensor): The weight tensor to be transformed.
  
  name (str): The name of the weight tensor as it appears in the state dict.

Returns:

  torch.Tensor: The transformed weight tensor.

"""


ShardActivationFn = Callable[[torch.nn.Module, str], torch.nn.Module]
"""
ShardActivationFn optionally transforms a module based on its name.

Args:

  module (torch.nn.Module): The module to be transformed.
  
  name (str): The name of the module as it appears in the state dict.

Returns:

  torch.nn.Module: The transformed module, or the original module if
    no transformation is needed.

"""


def shard_model(
  model: torch.nn.Module,
  shard_weight: ShardWeightFn,
  shard_activation: ShardActivationFn,
) -> torch.nn.Module:
  """
  Transforms `model` by applying `shard_weight` to each weight tensor and applying
  `shard_activation` to each module. Returns the transformed module.
  """
  state_dict = {}
  for name, param in model.state_dict().items():
    state_dict[name] = shard_weight(param, name)
  model.load_state_dict(state_dict, assign=True)
  return _wrap_module(model, shard_activation, tuple())


def _wrap_module(
  mod: torch.nn.Module, shard_activation: ShardActivationFn, prefix: tuple[str, ...]
) -> torch.nn.Module:
  """
  Recursively transforms the modules to shard activations.

  Start from the leaf modules and work our way up, to handle cases where one
  module is the child of another. The child modules will be transformed first,
  and then the parent module will be transformed, possibly with transformed
  children.
  """
  new_children = {}
  for name, child in mod.named_children():
    new_children[name] = _wrap_module(child, shard_activation, prefix + (name,))
  for name, new_child in new_children.items():
    mod.set_submodule(name, new_child)
  return shard_activation(mod, ".".join(prefix))


# TODO(https://github.com/AI-Hypercomputer/torchprime/issues/86): Add framework specific wrappers.


def shard_model_from_config(
  model: torch.nn.Module,
  config: dict,
  shard_output: Callable[[torch.Tensor, tuple[str, ...]], torch.Tensor],
  shard_param: Callable[[torch.Tensor, tuple[str, ...]], torch.Tensor] | None = None,
) -> torch.nn.Module:
  """
  Given a config of pattern to partition spec, shard the model accordingly.

  Example:

    ```python
    config = {
      # Shard the embedding projection
      'model.embed_tokens.weight': ['fsdp', None],
      # Shard the self-attention query projection
      'model.layers.*.self_attn.q_proj.weight': ['fsdp', None],
      # Shard the decoder layer outputs
      'model.layers.*': ['fsdp', None, None],
      # An empty string matches the output of the entire module.
      '': ['fsdp', None, None],
    }

    model = shard_model_from_config(model, config, xs.mark_sharding)
    ```

  A pattern may have an asterisk, which matches all immediate children whose
  name is an integer. This is useful for sharding all layers in a model.

  If a pattern matches a model parameter, then the parameter will be sharded.
  If a pattern matches a module, then the output of the module will be sharded.
  """

  if shard_param is None:
    shard_param = shard_output

  seen_params = set()
  seen_modules = set()

  def shard_weight(param, name):
    name = _process_sharding_name(name)
    spec = config.get(name)
    if spec is not None:
      seen_params.add(name)
      return shard_param(param, tuple(spec))
    return param

  def shard_activation(mod, name):
    name = _process_sharding_name(name)
    spec = config.get(name)
    if spec is not None:
      seen_modules.add(name)
      return ShardedModule(mod, shard_output, tuple(spec))
    return mod

  model = shard_model(model, shard_weight, shard_activation)

  want_names = set(config.keys())
  seen_names = seen_params.union(seen_modules)
  diff = "\n".join(want_names - seen_names)
  assert (
    seen_names == want_names
  ), f"""Requested to shard these names: {want_names}, but only sharded these: {seen_names}.
  
These names were not found in the model: {diff}.
"""

  return model


def _process_sharding_name(name):
  """Replace integers in param name with *."""

  def is_integer(t):
    try:
      int(t)
      return True
    # pylint: disable-next=all
    except:  # noqa: E722
      return False

  tokens = name.split(".")
  for i, t in enumerate(tokens):
    if is_integer(t):
      tokens[i] = "*"
  return ".".join(tokens)


class ShardedModule(torch.nn.Module):
  """
  Wraps an existing module and marks its output as sharded.
  """

  def __init__(self, mod, mark_sharding, spec):
    super().__init__()
    self._orig_mod = mod
    self.mark_sharding = mark_sharding
    self.spec = spec

  def forward(self, *args, **kwargs):
    return self.mark_sharding(self._orig_mod(*args, **kwargs), self.spec)
