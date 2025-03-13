# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Mixtral model."""
import jax
from jax.sharding import PartitionSpec as P

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from omegaconf import DictConfig
from torch import nn
from torch.nn import init


class MixtralBlockSparseTop2MLP(nn.Module):
  def __init__(self, intermediate_size, hidden_size):
    super().__init__()
    self.ffn_dim = intermediate_size
    self.hidden_dim = hidden_size

    self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
    self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
    self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    self.act_fn = F.silu

  def forward(self, hidden_states):
    current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
    current_hidden_states = self.w2(current_hidden_states)
    return current_hidden_states


class MixtralExpertCapacityTop2MLP(nn.Module):
  def __init__(self, intermediate_size, hidden_size, num_local_experts):
    super().__init__()
    self.ffn_dim = intermediate_size
    self.hidden_dim = hidden_size
    self.num_experts = num_local_experts

    self.w1 = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim))
    self.w2 = nn.Parameter(torch.empty(self.num_experts, self.ffn_dim, self.hidden_dim))
    self.w3 = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim))

    self.act_fn = F.silu

    init.kaiming_uniform_(self.w1, a=math.sqrt(5))
    init.kaiming_uniform_(self.w2, a=math.sqrt(5))
    init.kaiming_uniform_(self.w3, a=math.sqrt(5))

  def forward(self, dispatch_input):
    layer_w1 = torch.einsum("ebcm,emh->ebch", dispatch_input, self.w1)
    #xs.mark_sharding(layer_w1, mesh, ("expert", ("data", "fsdp"), None, None))
    # layer_w1.shard_(P("expert", ("data", "fsdp"), None, None))
    layer_w3 = torch.einsum("ebcm,emh->ebch", dispatch_input, self.w3)
    #xs.mark_sharding(layer_w3, mesh, ("expert", ("data", "fsdp"), None, None))
    # layer_w3.shard_(P("expert", ("data", "fsdp"), None, None))
    layer_multiply = self.act_fn(layer_w1) * layer_w3
    intermediate_layer = torch.einsum("ebch,ehm->ebcm", layer_multiply, self.w2)
    #xs.mark_sharding(intermediate_layer, mesh, ("expert", ("data", "fsdp"), None, None))
    # intermediate_layer.shard_(P("expert", ("data", "fsdp"), None, None))
    # TODO(bbahl): checkpoint intermediate_layer
    return intermediate_layer


class Gmm(torch.autograd.Function):
  @staticmethod
  def _eager_gmm(
    lhs: torch.Tensor, rhs: torch.Tensor, group_sizes: torch.Tensor
  ) -> torch.Tensor:
    """
    For testing purpose.
    """
    start = 0
    out = []
    for i, size in enumerate(group_sizes):
      result = lhs[start : start + size, :] @ rhs[i, :, :]
      out.append(result)
      start += group_sizes[i]
    return torch.cat(out)

  @staticmethod
  def _eager_gmm_backward(grad_output, lhs, rhs, group_sizes):
    """
    For testing purpose.
    """
    grad_lhs = []
    grad_rhs = []
    start = 0
    for i, size in enumerate(group_sizes):
      grad_lhs.append(
        grad_output[start : start + size, :] @ rhs[i, :, :].transpose(-1, -2)
      )
      grad_rhs.append(
        lhs[start : start + size, :].t() @ grad_output[start : start + size, :]
      )
      start += size
    return torch.cat(grad_lhs), torch.stack(grad_rhs)

def gmm_pallas(
    hidden_states: torch.Tensor,
    top_ks: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> torch.Tensor:
    """
    Integrated with PyTorch/XLA Pallas gmm:

    lhs: [m, hidden_size]
    top_ks: [m, k]
    w1: [num_experts, hidden_size, ffn_dim]
    w2: [num_experts, ffn_dim, hidden_size]
    w3: [num_experts, hidden_size, ffn_dim]
    """
    from torch_xla.experimental.custom_kernel import _histogram

    device = hidden_states.device
    if device == torch.device("cpu"):
      gmm = Gmm._eager_gmm
    # m is global shape
    m, k, n, num_experts, ffn_dim = (
      hidden_states.shape[0],
      top_ks.shape[1],
      hidden_states.shape[-1],
      w1.shape[0],
      w1.shape[-1],
    )

    # Create a new node to keep the original sharding spec.
    zero = torch.zeros((1,), device=device, dtype=hidden_states.dtype)
    full_w1 = w1 + zero
    full_w2 = w2 + zero
    full_w3 = w3 + zero

    # We want to create one big batch of tokens that has all top-k choices in it.
    # Our tokens will thus be duplicated k-times in the batch. To do this we,
    # first flatten the expert choices list and argsort it. This gives us an array
    # of length B * K. We then create a tiled arange of size B * K and index
    # into the expert choices list. This will give us the set of indices we need
    # to gather from the xs to create this big batch.
    top_flat = top_ks.flatten()
    hidden_states_order = top_flat.argsort()
    hidden_states_reverse_order = hidden_states_order.argsort()
    # Always replicated, so okay to skip manual sharding.
    hidden_states_indices = torch.arange(
      hidden_states.shape[0], device=device
    ).repeat_interleave(k)[hidden_states_order]
    hidden_states_sorted = hidden_states[hidden_states_indices]

    group_sizes = _histogram(top_flat.to(torch.int32), 0, num_experts - 1)
    # TODO(hanq): replace with torchax variants
    gmm1 = gmm(hidden_states_sorted, w1, group_sizes, tiling=(512, 1024, 1024))
    gmm3 = gmm(hidden_states_sorted, w3, group_sizes, tiling=(512, 1024, 1024))
    silu = F.silu(gmm1)
    sgmm = silu * gmm3
    gmm2 = gmm(sgmm, w2, group_sizes, tiling=(512, 1024, 1024))
    current_hidden_states = gmm2[hidden_states_reverse_order].reshape(-1, k, n)



class MixtralMoeBlock(nn.Module):
  def __init__(self, intermediate_size, hidden_size, num_local_experts, num_experts_per_tok, capacity_factor):
    super().__init__()
    self.hidden_dim = hidden_size
    self.ffn_dim = intermediate_size
    self.num_experts = num_local_experts
    self.top_k = num_experts_per_tok

    # Possible options are gmm, gmm_stack, dropping and static.
    # Huggingface native only implements static.
    self.moe_implementation = 'dropping' #config.moe_implementation

    # gating
    self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

    # initialize experts based on moe implementation
    match self.moe_implementation:
      case "gmm":
        assert False, "gmm"
      case "dropping":
        self.experts = MixtralExpertCapacityTop2MLP(
          intermediate_size=intermediate_size, 
          hidden_size=hidden_size, 
          num_local_experts=num_local_experts)
        # Only used for dropping implementation
        self.capacity_factor = capacity_factor
      case _:
        # gmm_stack and static initialize weights the same way.
        self.experts = nn.ModuleList(
          [MixtralBlockSparseTop2MLP(intermediate_size=intermediate_size, hidden_size=hidden_size) 
           for _ in range(self.num_experts)]
        )

  def generate_masks(self, top_k_indices, softmax_probs):
    """Generate masks to dispatch tokens to experts and combine moe activations afterwards.

    Only used for dropping implementation.
    """
    # calculate expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    batch_size, seq_len, _ = top_k_indices.shape
    tokens_per_batch = seq_len * self.top_k
    expert_capacity_per_batch = int(
      (tokens_per_batch / self.num_experts) * self.capacity_factor
    )
    print(
      f"Applying potential token dropping with a batch expert_capacity of {expert_capacity_per_batch}"
    )

    # calculate expert mask and drop tokens if needed
    # shape of output expert mask: (batch, sequence, num_experts_per_tok, num_experts)
    expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).to(torch.int32)
    expert_mask_fused = expert_mask.view(
      batch_size, seq_len * self.top_k, self.num_experts
    )  # (batch, s * top_k, e)
    #expert_mask_fused.shard_(P(("data", "fsdp", "expert"), None, None))
    expert_token_count_fused = torch.cumsum(
      expert_mask_fused, dim=1
    )  # (b, s * top_k , e)
    expert_token_count = expert_token_count_fused.view(
      batch_size, seq_len, self.top_k, self.num_experts
    )  # (b, s, k, e)
    #xs.mark_sharding(
    #  expert_token_count, mesh, (("data", "fsdp", "expert"), None, None, None)
    #)
    # expert_token_count.shard_(P(("data", "fsdp", "expert"), None, None, None))

    trunc_expert_mask = expert_mask * (
      expert_token_count <= expert_capacity_per_batch
    ).to(torch.int32)  # (b, s, k, e)
    combined_expert_mask = trunc_expert_mask.sum(dim=2)  # (b, s, e)

    # reshape & update weights
    softmax_probs = softmax_probs * combined_expert_mask  # (b, s, e)

    # calculate token position in expert capacity dimension
    expert_token_position_fused = (
      expert_mask_fused * expert_token_count_fused
    )  # (b, s, k, e)
    expert_token_position = expert_token_position_fused.view(
      batch_size, seq_len, self.top_k, self.num_experts
    )  # (b, s, k, e)
    combined_expert_token_position = (
      expert_token_position.sum(dim=2) * combined_expert_mask
    )  # (b, s, e)

    expert_token_position_in_capacity = F.one_hot(
      combined_expert_token_position, num_classes=expert_capacity_per_batch + 1
    ).to(torch.int32)  # (b, s, e, c)

    # shape of combine_mask is (batch_size, seq_len, num_experts, expert_capacity_per_batch + 1),
    # and cut 0-dimension which is always 0
    combine_mask = softmax_probs.unsqueeze(-1) * expert_token_position_in_capacity
    combine_mask = combine_mask[..., 1:]
    dispatch_mask = combine_mask.bool()  # (b, s, e, c)

    return dispatch_mask, combine_mask

  def load_balance_loss(self, top_k_indices, logits):
    """Additional loss to ensure tokens are equally distributed between experts.

    Only used when dropping implementation is used.
    Reference Switch Transformer https://arxiv.org/pdf/2101.03961
    """
    expert_mask = torch.nn.functional.one_hot(
      top_k_indices, num_classes=self.num_experts
    ).to(torch.int32)
    summed_expert_mask = torch.sum(expert_mask, dim=2)
    # Get fraction of tokens dispatched to each expert
    density = torch.mean(summed_expert_mask.float(), dim=1)  # Convert to float for mean
    # get fraction of probability allocated to each expert
    density_prob = torch.mean(logits, dim=1)
    loss = torch.mean(density * density_prob) * (self.num_experts**2)
    return loss

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    expert_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(expert_weights, self.top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)
    loss = 0.0
    match self.moe_implementation:
      case "static":
        final_hidden_states = torch.zeros(
          (batch_size * sequence_length, hidden_dim),
          dtype=hidden_states.dtype,
          device=hidden_states.device,
        )
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
          expert_layer = self.experts[expert_idx]
          routing_weights_idx = routing_weights.masked_fill(
            selected_experts != expert_idx, 0.0
          ).sum(dim=-1, keepdim=True)
          current_hidden_states = (
            expert_layer(hidden_states) * routing_weights_idx
          )  # We can't mask the input as there is non-linearities in the expert layer.
          final_hidden_states += current_hidden_states.to(hidden_states.dtype)
        final_hidden_states = final_hidden_states.reshape(
          batch_size, sequence_length, hidden_dim
        )
      case "dropping":
        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
        selected_experts = selected_experts.view(
          batch_size, sequence_length, self.top_k
        )
        expert_weights = expert_weights.view(
          batch_size, sequence_length, self.num_experts
        )
        dispatch_mask, combine_mask = self.generate_masks(
          selected_experts, expert_weights
        )
        combine_mask = combine_mask.to(hidden_states.dtype)
        mask_axes = (("data", "fsdp", "expert"), None, None, None)
        # xs.mark_sharding(dispatch_mask, mesh, mask_axes)
        # dispatch_mask.shard_(P(mask_axes))
        # xs.mark_sharding(combine_mask, mesh, mask_axes)
        # combine_mask.shard_(P(mask_axes))

        loss = self.load_balance_loss(selected_experts, expert_weights)
        # xs.mark_sharding(hidden_states, mesh, (("data", "fsdp", "expert"), None, None))
        # hidden_states.shard_(P(("data", "fsdp", "expert"), None, None))

        with jax.named_scope("bsm,bsec->ebcm"):
          dispatch = torch.einsum("bsm,bsec->ebcm", hidden_states, dispatch_mask)
        #xs.mark_sharding(dispatch, mesh, ("expert", ("data", "fsdp"), None, None))
        # dispatch.shard_(P("expert", ("data", "fsdp"), None, None))
        expert_layer = self.experts(dispatch)
        with jax.named_scope("ebcm,bsec -> bsm"):
          final_hidden_states = torch.einsum(
            "ebcm,bsec -> bsm", expert_layer, combine_mask
          )
        # xs.mark_sharding(
        #   final_hidden_states, mesh, (("data", "fsdp", "expert"), None, None)
        # )
        # final_hidden_states.shard_(P(("data", "fsdp", "expert"), None, None))
      case "gmm_stack":
        w1 = torch.stack([expert.w1.weight.t() for expert in self.experts])
        w2 = torch.stack([expert.w2.weight.t() for expert in self.experts])
        w3 = torch.stack([expert.w3.weight.t() for expert in self.experts])
        final_hidden_states = Gmm.apply(hidden_states, selected_experts, w1, w2, w3)
        final_hidden_states = (final_hidden_states * routing_weights[..., None]).sum(
          dim=1
        )
        final_hidden_states = final_hidden_states.reshape(
          batch_size, sequence_length, hidden_dim
        )
      case "gmm":
        final_hidden_states = self.experts(hidden_states, selected_experts)
        final_hidden_states = (final_hidden_states * routing_weights[..., None]).sum(
          dim=1
        )
        final_hidden_states = final_hidden_states.reshape(
          batch_size, sequence_length, hidden_dim
        )
    return final_hidden_states#, router_logits, loss