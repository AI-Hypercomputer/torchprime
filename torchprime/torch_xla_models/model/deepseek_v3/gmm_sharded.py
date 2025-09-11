from typing import Literal

import torch
import torch_xla.distributed.spmd as xs
from torch_xla.experimental.custom_kernel import _shard_map, gmm, tgmm

# --- Sharding specs specific to the DeepseekV3MoE layer ---
# These specs define how tensors are partitioned across the 3D device mesh.
TOKEN_PARTITION_SPEC = (("data", "fsdp"), "tensor")
W_GATE_UP_PARTITION_SPEC = ("expert", "tensor", "fsdp")
W_DOWN_PARTITION_SPEC = ("expert", "fsdp", "tensor")
# group_sizes is a 1D tensor, replicated across all devices.
GROUP_SIZES_PARTITION_SPEC = (None,)


def _gmm_forward_single_device(lhs, rhs, group_sizes, tiling):
  """The raw, single-device GMM forward operation."""
  return gmm(lhs, rhs, group_sizes, tiling)


def _gmm_backward_single_device(grad_output, lhs, rhs, group_sizes, tiling):
  """The raw, single-device GMM backward operation (gradient calculation)."""
  grad_lhs = gmm(grad_output, rhs.transpose(-1, -2), group_sizes, tiling)
  grad_rhs = tgmm(lhs.t(), grad_output, group_sizes, tiling)
  return grad_lhs, grad_rhs


class DeepseekMoEGMM(torch.autograd.Function):
  """
  A specialized, sharding-aware GMM wrapper for the DeepseekV3 MoE layer.

  It uses the `_shard_map` utility to correctly handle distributed execution by
  wrapping the single-device GMM kernels with the appropriate input and output
  sharding specifications. This mirrors the robust sharding pattern used in
  optimized kernels like FlashAttention in your project.
  """

  @staticmethod
  def forward(
    ctx,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    weight_type: Literal["gate_up", "down"],
    tiling: tuple[int, int, int] | None = None,
  ):
    mesh = xs.get_global_mesh()
    assert mesh is not None, "A global mesh must be defined to use DeepseekMoEGMM."

    if weight_type == "gate_up":
      rhs_partition_spec = W_GATE_UP_PARTITION_SPEC
    elif weight_type == "down":
      rhs_partition_spec = W_DOWN_PARTITION_SPEC
    else:
      raise ValueError(f"Invalid weight_type: {weight_type}")

    input_specs = [
      TOKEN_PARTITION_SPEC,
      rhs_partition_spec,
      GROUP_SIZES_PARTITION_SPEC,
      None,  # Tiling is a static argument, not a sharded tensor
    ]
    output_specs = [TOKEN_PARTITION_SPEC]

    # Create a sharded version of the forward function
    gmm_forward_callable = _shard_map(
      _gmm_forward_single_device, mesh, input_specs, output_specs
    )
    output = gmm_forward_callable(lhs, rhs, group_sizes, tiling)

    ctx.save_for_backward(lhs, rhs, group_sizes)
    ctx.tiling = tiling
    ctx.rhs_partition_spec = rhs_partition_spec

    return output

  @staticmethod
  def backward(ctx, grad_output: torch.Tensor):
    """Performs the sharded backward pass to compute gradients."""
    lhs, rhs, group_sizes = ctx.saved_tensors
    tiling = ctx.tiling
    rhs_partition_spec = ctx.rhs_partition_spec

    mesh = xs.get_global_mesh()
    assert mesh is not None, "A global mesh must be defined for backward pass."

    backward_input_specs = [
      TOKEN_PARTITION_SPEC,  # grad_output
      TOKEN_PARTITION_SPEC,  # lhs
      rhs_partition_spec,  # rhs
      GROUP_SIZES_PARTITION_SPEC,  # group_sizes
      None,  # tiling
    ]
    backward_output_specs = [
      TOKEN_PARTITION_SPEC,
      rhs_partition_spec,
    ]  # grad_lhs, grad_rhs

    # Create a sharded version of the backward function
    gmm_backward_callable = _shard_map(
      _gmm_backward_single_device, mesh, backward_input_specs, backward_output_specs
    )
    grad_lhs, grad_rhs = gmm_backward_callable(
      grad_output, lhs, rhs, group_sizes, tiling
    )

    return grad_lhs, grad_rhs, None, None, None
