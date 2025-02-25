from typing import Sequence
import torch.fx as fx
import torch
import torch_xla
from torch.utils._pytree import tree_iter

from functorch.compile import aot_function, make_boxed_func  # type:ignore
from .remat_all import remat_all_partition_fn


@torch.library.custom_op("xla::offload_name", mutates_args=())
def offload_name(t: torch.Tensor, name: str) -> torch.Tensor:
  """
  `offload_name` is an identity function that associates the input
  tensor with `name`. It is primarily useful in conjunction with
  `remat_all_and_offload_these_inputs`, which will rematerialize
  intermediate activations and also offload inputs with the specified
  names to host memory, moving them back during the backward pass.
  """
  if t is None:
    return None
  return t.clone()


@offload_name.register_fake
def _(t: torch.Tensor, name: str) -> torch.Tensor:
  if t is None:
    return None
  return torch.empty_like(t)


def offload_name_backward(ctx, grad):
  return grad, None


offload_name.register_autograd(offload_name_backward)


def remat_all_and_offload_these_inputs(
    joint_module: fx.GraphModule,
    _joint_inputs,
    *,
    num_fwd_outputs,
    names_to_offload: Sequence[str],
):
  """
  `remat_all_and_offload_these_inputs` will rematerialize (recompute) all
  intermediate activations in `joint_module`, and offload inputs with the
  specified names to host memory, moving them back during the backward pass.
  It transforms the joint graph into separate forward and backward graphs.
  """
  input_device = next(iter(tree_iter(_joint_inputs))).device
  fwd, bwd = remat_all_partition_fn(
      joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs)
  with torch.device(input_device):
    fw_example_args = make_arguments(fwd)
    bw_example_args = make_arguments(bwd)

  fw_name_in_output_indices = get_name_in_output_indices(fwd)
  bw_name_in_input_names = get_name_in_input_names(bwd)

  for name in names_to_offload:
    print(f"Going to offload {name}")
    assert name in fw_name_in_output_indices
    assert name in bw_name_in_input_names

  # print("fw_name_in_output_indices", fw_name_in_output_indices)
  # print("bw_name_in_input_names", bw_name_in_input_names)

  with torch.no_grad():

    def forward(**kwargs):
      import pdb
      try:
        out = fwd(**kwargs)
        indices_to_offload = set(
            [fw_name_in_output_indices[name] for name in names_to_offload])
        return tuple(
            torch.ops.xla.place_to_host(v) if i in  # type:ignore
            indices_to_offload else v for i, v in enumerate(out))
      except Exception:
        pdb.post_mortem()

    def backward(**kwargs):
      # print(f"Backward got {len(kwargs)} arguments:")
      for k, v in kwargs.items():
        print(f"Arg {k}: {v.shape if v is not None else 'None'}")
      arguments_to_move_back = set(
          [bw_name_in_input_names[name] for name in names_to_offload])
      kwargs = {
          k: torch.ops.xla.place_to_device(v)  # type: ignore
          if k in arguments_to_move_back else v for k, v in kwargs.items()
      }
      import pdb
      try:
        values = bwd(**kwargs)
        print(f"Backward will return {len(values)} values:")
        for i, v in enumerate(values):
          print(f"Arg {i}: {v.shape if v is not None else 'None'}")
        return values
      except Exception:
        pdb.post_mortem()

    # Use AOTAutograd to retrace forward and backward, thus incorporating
    # the offloading ops.
    graph = [None]

    def get_graph(g, _):
      graph[0] = g
      return make_boxed_func(g)

    _ = aot_function(forward, fw_compiler=get_graph)(**fw_example_args)
    aot_forward = graph[0]

    _ = aot_function(backward, fw_compiler=get_graph)(**bw_example_args)
    aot_backward = graph[0]

    return aot_forward, aot_backward


def make_arguments(gm: fx.GraphModule):
  """
  Given a graph module, `make_arguments` returns a dictionary of example inputs
  that can be used as keyward arguments to call the graph module.
  """
  example_args = {}
  for node in gm.graph.nodes:
    if node.op != 'placeholder':
      continue
    if 'tensor_meta' in node.meta:
      tensor_meta = node.meta['tensor_meta']
      tensor = torch.zeros(
          tensor_meta.shape,
          dtype=tensor_meta.dtype,
          requires_grad=tensor_meta.requires_grad)
      example_args[node.name] = tensor
  return example_args


def get_named_nodes(gm: torch.fx.GraphModule):
  named_nodes = {}

  for node in gm.graph.nodes:
    if node.op == "call_function":
      if hasattr(node.target, "name"):
        if node.target.name() == offload_name._qualname:  # type: ignore
          named_nodes[node.args[0]] = node.args[1]

  return named_nodes


def get_name_in_output_indices(gm: torch.fx.GraphModule):
  named_nodes = get_named_nodes(gm)
  name_in_output_indices = {}

  for node in gm.graph.nodes:
    if node.op == "output":
      assert len(node.args) <= 1
      if len(node.args) == 0:
        continue
      for i, arg in enumerate(next(iter(node.args))):  # type: ignore
        if arg in named_nodes:
          name_in_output_indices[named_nodes[arg]] = i

  return name_in_output_indices


def get_name_in_input_names(gm: torch.fx.GraphModule):
  named_nodes = get_named_nodes(gm)
  name_in_input_names = {}

  for node in gm.graph.nodes:
    if node.op == "placeholder":
      if node in named_nodes:
        name_in_input_names[named_nodes[node]] = node.target

  return name_in_input_names