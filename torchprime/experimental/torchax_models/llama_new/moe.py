# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# ruff: noqa: N806
# pyre-strict
from typing import Optional

import torch

from torch import nn, Tensor
from torch.nn import functional as F

from .args import MoEArgs


def ColumnParallelLinear(out_dim, indim, bias, gather_output=True, init_method=None):
    return torch.nn.Linear(out_dim, indim, bias)


def RowParallelLinear(out_dim, indim, bias, input_is_parallel=True, init_method=None):
    return torch.nn.Linear(out_dim, indim, bias)


def VocabParallelEmbedding(vocab_size, args, init_method):
    return torch.nn.Embedding(vocab_size, args)


class Experts(nn.Module):
    def __init__(
        self,
        num_local_experts: int,
        dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        dtype = torch.get_default_dtype()
        self.num_local_experts = num_local_experts
        divide_factor = 1

        self.moe_w_in_eD_F: nn.Parameter = nn.Parameter(
            torch.empty(
                num_local_experts * dim,
                divide_exact(hidden_dim, divide_factor),
                dtype=dtype,
            )
            if num_local_experts > 1
            else torch.empty(dim, divide_exact(hidden_dim, divide_factor), dtype=dtype)
        )

        self.moe_w_out_eF_D: nn.Parameter = nn.Parameter(
            torch.empty(
                num_local_experts * divide_exact(hidden_dim, divide_factor),
                dim,
                dtype=dtype,
            )
            if num_local_experts > 1
            else torch.empty(divide_exact(hidden_dim, divide_factor), dim, dtype=dtype)
        )

        self.moe_w_swiglu_eD_F: nn.Parameter = nn.Parameter(
            torch.empty(
                num_local_experts * dim,
                divide_exact(hidden_dim, divide_factor),
                dtype=dtype,
            )
            if num_local_experts > 1
            else torch.empty(dim, divide_exact(hidden_dim, divide_factor), dtype=dtype)
        )

    def forward(
        self,
        routed_in_egD: torch.Tensor,
    ) -> torch.Tensor:
        moe_w_in_eD_F = self.moe_w_in_eD_F
        moe_w_swiglu_eD_F = self.moe_w_swiglu_eD_F
        moe_w_out_eF_D = self.moe_w_out_eF_D

        return self._expert_function(
            routed_in_egD,
            moe_w_in_eD_F,
            moe_w_swiglu_eD_F,
            moe_w_out_eF_D,
        )

    def _expert_function(
        self,
        x: torch.Tensor,
        w_in_eDF: torch.Tensor,
        w_swiglu_eDF: torch.Tensor,
        w_out_eFD: torch.Tensor,
    ) -> torch.Tensor:
        eD, _ = w_in_eDF.shape
        e = self.num_local_experts
        D = divide_exact(eD, self.num_local_experts)

        w_in_eDF = w_in_eDF.view(e, D, -1)
        w_swiglu_eDF = w_swiglu_eDF.view(e, D, -1)
        w_out_eFD = w_out_eFD.view(e, -1, D)

        x_egD = x
        x_egD = x_egD.view(e, -1, D)

        middle_egF = torch.bmm(x_egD, w_in_eDF) # x1
        swiglu_hidden_egF = torch.bmm(x_egD, w_swiglu_eDF) #x3
        middle_out_egF = F.silu(middle_egF) * swiglu_hidden_egF

        out_egD = torch.bmm(middle_out_egF, w_out_eFD)
        out_egD = out_egD.view(-1, D)

        return out_egD


class MoE(torch.nn.Module):
    """
    This EC implementation is modified from the original EC module.
    We refactored the token permutation and unpermutation logic and added support to tp and dp2ep sharding.
    This module supports 3 sharding methods of the experts:
    - tp: each TP rank has n_experts experts. Experts are sharded following the conventional row/column-parallel TP sharding.
    - tp2ep: each TP rank has n_experts/tp experts. Experts are not sharded.
    - dp2ep: each EP rank has n_experts/ep experts. Experts are sharded following the row/column-parallel TP sharding.
    Tensors used in this module are annotated with the suffixes that indicate the shape of the tensor.
    Several commonly used annotations include:
    - a: bsz*slen
    - E: number of experts
    - e: number of local experts per ep (n_experts/ep)
    - et: number of local experts per tp (n_experts/tp)
    - D: hidden dimension
    - d: D/tp
    - F: model dimension
    - f: F/tp (used in column/row-parallel linear)
    - G: number of tokens per expert (a * capacity_factor / E)
    - g: number of tokens per expert per TP rank (i.e., G/TP)
    - GG: G*EP (number of tokens per expert received via inter-EP a2a when ag_along_first_dim=False)
    - gg: g*EP (number of tokens per expert received via inter-EP a2a when ag_along_first_dim=True)

    Examples:
    x_aD [a, D]
    routed_in_etG_D [et*G, D]
    x_eGGD: [e, GG, D]
    """

    def __init__(
        self,
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
        dtype: torch.dtype = torch.get_default_dtype()
        self.experts = Experts(
            num_local_experts,
            dim,
            hidden_dim,
        )

        self.router_DE: nn.Parameter = nn.Parameter(torch.empty(dim, moe_args.num_experts, dtype=dtype))

        self.w_in_shared_FD: Optional[ColumnParallelLinear] = None
        self.w_out_shared_DF: Optional[RowParallelLinear] = None
        self.w_swiglu_FD: Optional[ColumnParallelLinear] = None
        if moe_args.use_shared_expert:
            self.w_in_shared_FD = ColumnParallelLinear(
                dim,
                hidden_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )

            self.w_out_shared_DF = RowParallelLinear(
                hidden_dim,
                dim,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: x,
            )
            self.w_swiglu_FD = ColumnParallelLinear(
                dim,
                hidden_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )

        if not moe_args.use_token_choice:
            # mean, var * count, count
            self.register_buffer(
                "global_gate_stats_3E",
                torch.zeros(3, moe_args.num_experts, dtype=torch.float32),
            )

    def forward(self, x_bsD: Tensor) -> Tensor:
        _, slen, D = x_bsD.shape
        x_aD = x_bsD.view(-1, D)
        assert self.w_in_shared_FD is not None
        assert self.w_out_shared_DF is not None
        assert self.w_swiglu_FD is not None

        threshold_E = self.global_gate_stats_3E[0] if not self.moe_args.use_token_choice else None
        out_aD = MoEFunction.forward(
            x_aD,
            self.router_DE,
            self.w_in_shared_FD.weight,
            self.w_out_shared_DF.weight,
            self.w_swiglu_FD.weight,
            self.experts,
            self.moe_args,
            threshold_E,
        )

        return out_aD.view(-1, slen, D)


class MoEFunction:
    @staticmethod
    def forward(
        x: Tensor,
        router_DE: Tensor,
        w_in_shared_FD: Tensor,
        w_out_shared_DF: Tensor,
        w_swiglu_FD: Tensor,
        experts: torch.nn.Module,
        moe_args: MoEArgs,
        expert_threshold_E: Optional[torch.Tensor] = None,
    ) -> Tensor:
        E: int = moe_args.num_experts
        x_aD: Tensor = x
        a, D = x_aD.shape
        
        # this is gate matmul
        router_scores_Ea: Tensor = torch.matmul(x_aD, router_DE).transpose(0, 1)
        tokens_per_expert: int = a

        if moe_args.use_token_choice:
            router_scores_aK, router_indices_aK = torch.topk(router_scores_Ea.transpose(0, 1), moe_args.top_k, dim=1)
            router_scores_EG = (
                torch.full_like(router_scores_Ea.transpose(0, 1), float("-inf"))
                .scatter_(1, router_indices_aK, router_scores_aK)
                .transpose(0, 1)
            )
            router_indices_EG = torch.arange(a, device=x_aD.device).view(1, -1).expand(router_scores_EG.size(0), -1)
        else:
            if expert_threshold_E is None:
                tokens_per_expert = int(a * moe_args.capacity_factor / E)
                tokens_per_expert += -tokens_per_expert % 8  # round to multiple of 8
                tokens_per_expert = min(tokens_per_expert, a)
                router_scores_EG, router_indices_EG = torch.topk(router_scores_Ea, tokens_per_expert, dim=1)
            else:
                router_scores_EG = torch.where(
                    router_scores_Ea >= expert_threshold_E.unsqueeze(1),
                    router_scores_Ea,
                    torch.full_like(router_scores_Ea, fill_value=float("-inf")),
                )

                router_indices_EG = torch.arange(a, device=x_aD.device).view(1, -1).expand(router_scores_EG.size(0), -1)

        router_scores_EG = torch.sigmoid(router_scores_EG.float()).to(x.dtype)

        routed_in_EG_D: Tensor = torch.gather(
            x_aD,
            dim=0,
            index=router_indices_EG.reshape(-1, 1).expand(-1, D),
        )
        routed_in_EG_D = routed_in_EG_D * router_scores_EG.reshape(-1, 1)

        if moe_args.use_shared_expert:
            shared_middle_out_aF = shared_expert_fc13(
                x_aD,
                w_in_shared_FD,
                w_swiglu_FD,
            )

        routed_out_egg_D = experts(routed_in_EG_D.detach())
        if moe_args.use_shared_expert:
            out_aD = shared_expert_fc2(
                shared_middle_out_aF,
                w_out_shared_DF,
            )
        else:
            out_aD = torch.zeros_like(x_aD)

        router_indices_EG_D = router_indices_EG.reshape(-1, 1).expand(-1, D)
        out_aD.scatter_add_(
            dim=0,
            index=router_indices_EG_D,
            src=routed_out_egg_D.view(-1, D),
        )
        #reduce_from_model_parallel_region(out_aD)
        return out_aD


@torch.no_grad()
def shared_expert_fc13(
    x_aD: torch.Tensor,
    w_in_shared_FD: torch.Tensor,
    w_swiglu_FD: torch.Tensor,
) -> Tensor:
    shared_middle_af = F.linear(x_aD, w_in_shared_FD)
    swiglu_hidden_af = F.linear(x_aD, w_swiglu_FD)
    return F.silu(shared_middle_af) * swiglu_hidden_af


@torch.no_grad()
def shared_expert_fc2(
    shared_middle_out_aF: Tensor,
    w_out_shared_DF: Tensor,
) -> Tensor:
    return F.linear(shared_middle_out_aF, w_out_shared_DF)


def divide_exact(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)
    return numerator // denominator
