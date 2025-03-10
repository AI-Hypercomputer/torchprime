from contextlib import contextmanager

import torch._functorch.config
import torch.fx
from functorch.compile import min_cut_rematerialization_partition


@contextmanager
def remat_all_config():
  # Backup existing config values
  backup = {
    "activation_memory_budget": torch._functorch.config.activation_memory_budget,
    "aggressive_recomputation": torch._functorch.config.aggressive_recomputation,
    "recompute_views": torch._functorch.config.recompute_views,
    "ban_recompute_reductions": torch._functorch.config.ban_recompute_reductions,
    "ban_recompute_not_in_allowlist": torch._functorch.config.ban_recompute_not_in_allowlist,
    "ban_recompute_materialized_backward": torch._functorch.config.ban_recompute_materialized_backward,
    "ban_recompute_long_fusible_chains": torch._functorch.config.ban_recompute_long_fusible_chains,
    "ban_recompute_used_far_apart": torch._functorch.config.ban_recompute_used_far_apart,
  }

  try:
    # Set activation_memory_budget to zero to force the min cut partitioner
    # to recompute instead of saving. Also don't ban the recomputing of any ops.
    torch._functorch.config.activation_memory_budget = 0.0
    torch._functorch.config.aggressive_recomputation = True
    torch._functorch.config.recompute_views = True
    torch._functorch.config.ban_recompute_reductions = False
    torch._functorch.config.ban_recompute_not_in_allowlist = False
    torch._functorch.config.ban_recompute_materialized_backward = False
    torch._functorch.config.ban_recompute_long_fusible_chains = False
    torch._functorch.config.ban_recompute_used_far_apart = False
    yield

  finally:
    # Restore the original config values
    torch._functorch.config.activation_memory_budget = backup[
      "activation_memory_budget"
    ]
    torch._functorch.config.aggressive_recomputation = backup[
      "aggressive_recomputation"
    ]
    torch._functorch.config.recompute_views = backup["recompute_views"]
    torch._functorch.config.ban_recompute_reductions = backup[
      "ban_recompute_reductions"
    ]
    torch._functorch.config.ban_recompute_not_in_allowlist = backup[
      "ban_recompute_not_in_allowlist"
    ]
    torch._functorch.config.ban_recompute_materialized_backward = backup[
      "ban_recompute_materialized_backward"
    ]
    torch._functorch.config.ban_recompute_long_fusible_chains = backup[
      "ban_recompute_long_fusible_chains"
    ]
    torch._functorch.config.ban_recompute_used_far_apart = backup[
      "ban_recompute_used_far_apart"
    ]


def remat_all_partition_fn(
  joint_module: torch.fx.GraphModule,
  _joint_inputs,
  *,
  num_fwd_outputs,
):
  """
  remat_all_partition_fn is a graph partition function that closely matches the
  default behavior of `torch.utils.checkpoint`, which is to discard all intermediate
  activations and recompute all of them during the backward pass.
  """
  with remat_all_config():
    return min_cut_rematerialization_partition(
      joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs
    )
