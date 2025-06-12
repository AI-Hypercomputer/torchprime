"""
Some pieces of the model code were copied from https://github.com/pytorch/torchrec/blob/main/torchrec/models/dlrm.py
#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""

import logging
import math

import torch
import torch.nn as nn
import torch_xla.core.xla_builder as xb
from omegaconf import DictConfig

from torchprime.experimental.torchax_models.dlrm.dist_utils import make_data_parallel
from torchprime.experimental.torchax_models.dlrm.interop_flax import (
  get_dense_embed_module,
  reduce_embeddings_jax,
)
from torchprime.experimental.torchax_models.dlrm.torcrec_stub import LowRankCrossNet

logger = logging.getLogger(__name__)


def choose(n: int, k: int) -> int:
  """
  Simple implementation of math.comb for Python 3.7 compatibility.
  """
  if 0 <= k <= n:
    ntok = 1
    ktok = 1
    for t in range(1, min(k, n - k) + 1):
      ntok *= n
      ktok *= t
      n -= 1
    return ntok // ktok
  else:
    return 0


class BaseInteract(nn.Module):
  def __init__(
    self,
    config: DictConfig,
    interact_indices,
  ):
    super().__init__()
    self.interact_indices = interact_indices

  def forward(self, dense_features: torch.Tensor, sparse_features: torch.Tensor):
    """
    Args:
        dense_features (torch.Tensor): an input tensor of size B X D.
        sparse_features (torch.Tensor): an input tensor of size B X F X D.
    Returns:
        torch.Tensor: an output tensor of size B X (D + F + F choose 2).
    """
    # concatenate dense and sparse features
    conc_features = torch.cat((dense_features.unsqueeze(1), sparse_features), dim=1)
    # perform a dot product between all pairs of features
    Z = torch.bmm(conc_features, torch.transpose(conc_features, 1, 2))
    interactions_flat = Z[:, self.interact_indices[0], self.interact_indices[1]]

    # finally, concatenate dense features and interactions
    R = torch.cat((dense_features, interactions_flat), dim=1)
    return R


class DCNv2Interact(nn.Module):
  def __init__(
    self,
    num_sparse_features: int,
    embedding_dim: int,
    dcn_num_layers: int,
    dcn_low_rank_dim: int,
  ):
    super().__init__()
    self.num_sparse_features = num_sparse_features
    self.crossnet = LowRankCrossNet(
      in_features=(num_sparse_features + 1) * embedding_dim,
      num_layers=dcn_num_layers,
      low_rank=dcn_low_rank_dim,
    )

  def forward(
    self, dense_features: torch.Tensor, sparse_features: torch.Tensor
  ) -> torch.Tensor:
    """
    Args:
        dense_features (torch.Tensor): an input tensor of size B X D.
        sparse_features (torch.Tensor): an input tensor of size B X F X D.
    Returns:
        torch.Tensor: an output tensor of size B X (F*D + D).
    """
    if self.num_sparse_features <= 0:
      return dense_features
    (B, _D) = dense_features.shape

    combined_values = torch.cat((dense_features.unsqueeze(1), sparse_features), dim=1)

    # size B X (F*D + D)
    return self.crossnet(combined_values.reshape([B, -1]))


class DlrmModel(nn.Module):
  @classmethod
  def from_cfg(cls, cfg: DictConfig):
    emb_module = get_dense_embed_module(cfg)
    return DlrmModel(cfg.model, emb_module)

  def __init__(
    self,
    cfg: DictConfig,
    emb_module: nn.Module,
  ):
    super().__init__()
    self.emb_module = emb_module
    embedding_dim = cfg.sparse_feature_size
    dense_arch_layer_sizes = cfg.dense_mlp.layers

    if dense_arch_layer_sizes[-1] != embedding_dim:
      raise ValueError(
        f"embedding_bag_collection dimension ({embedding_dim}) and final dense "
        "arch layer size ({dense_arch_layer_sizes[-1]}) must match."
      )

    # init dense module
    self.dense_arch = self.create_mlp(
      cfg.dense_features_num,
      dense_arch_layer_sizes,
      cfg.dense_mlp.activation,
    )

    # init interact module
    self.sparse_features_num = len(cfg.num_embeddings_per_feature)  # t
    total_features_num = len(cfg.num_embeddings_per_feature) + 1
    num_sparse_features = len(cfg.num_embeddings_per_feature)

    if cfg.interact_module.type != "dcnv2":  # base dot product
      interact_indices = torch.tril_indices(
        total_features_num, total_features_num, offset=-1
      )
      self.interact_module = BaseInteract(cfg, interact_indices)
      top_in_features = (
        embedding_dim + choose(num_sparse_features, 2) + num_sparse_features
      )
    else:
      self.interact_module = DCNv2Interact(
        num_sparse_features,
        embedding_dim,
        cfg.interact_module.dcnv2.num_layers,
        cfg.interact_module.dcnv2.low_rank_dim,
      )
      top_in_features: int = (num_sparse_features + 1) * embedding_dim

    # init top module
    top_arch_layer_sizes = cfg.top_mlp.layers

    self.top_arch = self.create_mlp(
      top_in_features,
      top_arch_layer_sizes,
      cfg.top_mlp.activation,
      activation_at_last=False,
    )
    self.do_butterfly_shuffle = cfg.do_butterfly_shuffle

  def create_mlp(
    self,
    input_dim: int,
    layers_dims: list[int],
    activation: str,
    activation_at_last: bool = True,
  ):
    # build MLP layer by layer
    layers = nn.ModuleList()
    layers_dims = [input_dim] + layers_dims
    for i in range(len(layers_dims) - 1):
      input_dim = layers_dims[i]
      out_dim = layers_dims[i + 1]
      layer = nn.Linear(layers_dims[i], layers_dims[i + 1])
      nn.init.normal_(layer.weight.data, 0.0, math.sqrt(2.0 / (input_dim + out_dim)))
      nn.init.normal_(layer.bias.data, 0.0, math.sqrt(1.0 / out_dim))
      layers.append(layer)
      if not activation_at_last and i == len(layers_dims) - 2:
        continue
      if activation == "relu":
        layers.append(nn.ReLU())
      elif activation == "sigmoid":
        layers.append(nn.Sigmoid())
    return torch.nn.Sequential(*layers)

  def apply_mlp_bot(self, x: torch.Tensor) -> torch.Tensor:
    return self.dense_arch(x)

  def apply_mlp_top(self, x: torch.Tensor) -> torch.Tensor:
    return self.top_arch(x)

  def apply_emb(
    self, features_values: torch.Tensor, features_lengths: torch.Tensor, bsz: int
  ) -> torch.Tensor:
    """
    Args:
      sparse features values: 1D of dynamic length
      sparse features length: 1D of length = B x F_N
    Returns:
      reduced embeddings tensor of shape: B x F_N x EMB_DIM
    """
    all_batch_embeddings = self.emb_module(features_values)

    # it needs to be converted into a tensor of size (B, F_N, embedding_dim)
    # summing emebddings for multi-hot features.
    embedding_dim = all_batch_embeddings.size(1)

    # temporal workaround & solution until we get sparse core & tensors support
    summed_embeddings_tensor = xb.call_jax(
      reduce_embeddings_jax, (all_batch_embeddings, features_lengths)
    )

    # pytorch version of the above which is currently not supported by torchax:
    # summed_embeddings_tensor = torch.segment_reduce(
    #   data=all_batch_embeddings, reduce="sum", lengths=features_lengths
    # )

    result_embeddings = summed_embeddings_tensor.view(
      bsz, self.sparse_features_num, embedding_dim
    )
    return result_embeddings

  def forward(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """
    Args:
      batch:
       dense features: B x D_N
       sparse features values: 1D of dynamic length
       sparse features length: 1D of length = B x F_N

      where D_N - num of dense features
      F_N - num of sparse features

    Retuns:
      prediction model logits: B x 1
    """
    dense_features = batch[0]
    bsz = dense_features.size(0)

    dense_emb = self.apply_mlp_bot(dense_features)
    sparse_emb = self.apply_emb(batch[1], batch[2], bsz)
    interacted_out: torch.Tensor = self.interact_module(dense_emb, sparse_emb)

    if self.do_butterfly_shuffle:
      interacted_out = make_data_parallel(interacted_out)  # type: ignore

    top_mlp_logits = self.apply_mlp_top(interacted_out)
    logits = top_mlp_logits.squeeze(-1)

    return logits
