# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
import jax
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .args import ModelArgs, MoEArgs
from .datatypes import CoreTransformerInput, CoreTransformerOutput
from .moe import MoE, ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding

from torchprime.experimental.torchax_models.mixtral_model import MixtralMoeBlock

from torchprime.experimental.torchax_models.llama.model_with_scan import ScanLayer


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class L2Norm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x)


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    # TODO: this module needs to be moved into a separate file since it can be used by
    # the vision encoder as well.
    def __init__(self, args: ModelArgs, add_bias: bool = False):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=add_bias,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=add_bias,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=add_bias,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=add_bias,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.qk_norm = L2Norm(args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        if self.qk_norm:
            xq = self.qk_norm(xq)
            xk = self.qk_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)


        xq, xk, xv = [t.transpose(1, 2) for t in (xq, xk, xv)]

        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        attn_output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, dropout_p=0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(attn_output)
        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

        

class ConditionalFeedForward(torch.nn.Module):

  def __init__(self, num_experts, hidden_dim, dim):
    super().__init__()
    self.w1 = nn.Parameter(
        torch.empty(num_experts, dim, hidden_dim)
    )
    self.w2 = nn.Parameter(
        torch.empty(num_experts, hidden_dim, dim)
    )
    self.w3 = nn.Parameter(
        torch.empty(num_experts, dim, hidden_dim)
    )

  def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
    seqlen = x.shape[0]

    # e = total num of exp = 16
    # t = seqlen
    # o = config.imtermediate size
    # i = config.dim
    with jax.named_scope("conditional_ff"):
      x1 = F.silu(torch.einsum("ti,eoi -> teo", x, self.w1))
      x3 = torch.einsum("ti, eoi-> teo", x, self.w3)
      expert_outs = torch.einsum("teo, eio -> tei", (x1 * x3), self.w2)
      # e = 16; need to reduce to 1
      seq_indexes = torch.arange(seqlen, device=x.device).unsqueeze(1)
      return expert_outs[seq_indexes, expert_indices]


class MOEFeedForward(torch.nn.Module):

  def __init__(self,
        dim: int,
        hidden_dim: int,
        ffn_dim_multiplier: float,
        multiple_of: int,
        moe_args: MoEArgs,
    ) -> None:
    super().__init__()
    self.moe_args = moe_args

    hidden_dim_denom: float = 1
    if moe_args.auto_scale_F:
        hidden_dim_denom = moe_args.capacity_factor + int(moe_args.use_shared_expert)

    hidden_dim = int(2 * hidden_dim / 3)

    # custom dim factor multiplier
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)

    if moe_args.auto_scale_F:
        hidden_dim = int(hidden_dim / hidden_dim_denom)

    # round hidden dimension to `multiple_of`
    hidden_dim += -hidden_dim % multiple_of

    # Sharding method is tp2ep: each TP rank has n_experts/tp experts. Experts are not sharded.
    num_local_experts: int = moe_args.num_experts

    self.shared_expert = FeedForward(
        dim=dim,
        hidden_dim=hidden_dim,
    )

    self.gated_experts = MixtralMoeBlock(
        intermediate_size=hidden_dim,
        hidden_size=dim,
        num_local_experts=num_local_experts,
        num_experts_per_tok=1,
        capacity_factor=moe_args.capacity_factor
    )

    # self.gate = torch.nn.Linear(dim, moe_args.num_experts)
    # self.cond_ffn = ConditionalFeedForward(num_local_experts, dim, hidden_dim)
    # self.dim = dim
    # self.num_activated_experts = moe_args.top_k

  def forward(self, x):
    experts_outs = self.gated_experts(x) 
    shared_exp_out = self.shared_expert(x)
    return experts_outs + shared_exp_out


#   def forward(self, x: torch.Tensor) -> torch.Tensor:
#     bsz, seq, hidden = x.shape
#     # [B, T, D], combine BT, for prefill B = 1, for decode, T = 1
#     x = x.view(-1, self.dim)
#     # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
#     # x: [T, D]
#     scores = self.gate(x)  # [T, E]
#     expert_weights = torch.sigmoid(scores)
#     # expert_weights = F.softmax(scores, dim=-1)
#     expert_weights, expert_indices = torch.topk(
#         expert_weights, self.num_activated_experts, dim=-1
#     )  # [T, A], [T, A]
#     #expert_weights /= expert_weights.sum(dim=-1, keepdim=True)  # [T, A]
#     expert_outs = self.cond_ffn(x, expert_indices)
#     expert_outs = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
#     # Changes back to [B, T, D]
#     expert_outs = expert_outs.reshape(bsz, seq, hidden)
#     shared_exp_out = self.shared_expert(x).reshape(bsz, seq, hidden)
#     return expert_outs + shared_exp_out



class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)

        self.feed_forward = MOEFeedForward(
            dim=args.dim,
            hidden_dim=int(args.ffn_exp * args.dim),
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            multiple_of=args.multiple_of,
            moe_args=args.moe_args,
        )

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class CoreTransformer(nn.Module):
    def __init__(self, args: ModelArgs, **kwargs) -> None:
        super().__init__()
        self.args = args

        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = VocabParallelEmbedding(args.vocab_size, args.dim, init_method=lambda x: x)

        self.layers = torch.nn.ModuleList()

        # for layer_id in range(args.n_layers):
        #     self.layers.append(TransformerBlock(layer_id, args))

        self.layers = ScanLayer(TransformerBlock(0, args), args.n_layers)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(args.dim, args.vocab_size, bias=False, init_method=lambda x: x)

        # self.freqs_cis = precompute_freqs_cis(
        #     args.dim // args.n_heads,
        #     args.max_seq_len * 2,
        #     args.rope_theta,
        #     args.use_scaled_rope,
        # )
        # vision_args = self.args.vision_args
        if False: #vision_args:
            # circular import otherwise until we refactor out Attention
            from .vision.embedding import VisionEmbeddings

            self.vision_embeddings = VisionEmbeddings(vision_args)
            self.vision_projection = ColumnParallelLinear(
                vision_args.output_dim,
                args.dim,
                bias=False,
                init_method=lambda x: x,
            )

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if prefix + "rope.freqs" in state_dict:
            state_dict.pop(prefix + "rope.freqs")

    def forward(self, tokens, start_pos, freqs_cis, mask):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        # for layer in self.layers:
        #     h = layer(h, start_pos, freqs_cis, mask)
        h = self.layers(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output

