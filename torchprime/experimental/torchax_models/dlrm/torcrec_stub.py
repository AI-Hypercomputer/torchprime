"""
CrossNet Components from torchrec.
# Taken from here https://github.com/pytorch/torchrec/blob/main/torchrec/modules/crossnet.py#L94
"""

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


class LowRankCrossNet(torch.nn.Module):
  r"""
  Low Rank Cross Net is a highly efficient cross net. Instead of using full rank cross
  matrices (NxN) at each layer, it will use two kernels :math:`W (N x r)` and
  :math:`V (r x N)`, where `r << N`, to simplify the matrix multiplication.

  On each layer l, the tensor is transformed into:

  .. math::    x_{l+1} = x_0 * (W_l \cdot (V_l \cdot x_l) + b_l) + x_l

  where :math:`W_l` is either a vector, :math:`*` means element-wise multiplication,
  and :math:`\cdot` means matrix multiplication.

  NOTE:
      Rank `r` should be chosen smartly. Usually, we  expect `r < N/2` to have
      computational savings; we should expect :math:`r ~= N/4` to preserve the
      accuracy of the full rank cross net.

  Args:
      in_features (int): the dimension of the input.
      num_layers (int): the number of layers in the module.
      low_rank (int): the rank setup of the cross matrix (default = 1).
          Value must be always >= 1.

  Example::

      batch_size = 3
      num_layers = 2
      in_features = 10
      input = torch.randn(batch_size, in_features)
      dcn = LowRankCrossNet(num_layers=num_layers, low_rank=3)
      output = dcn(input)
  """

  def __init__(
    self,
    in_features: int,
    num_layers: int,
    low_rank: int = 1,
  ) -> None:
    super().__init__()
    assert low_rank >= 1, "Low rank must be larger or equal to 1"

    self._num_layers = num_layers
    self._low_rank = low_rank
    self.W_kernels: torch.nn.ParameterList = torch.nn.ParameterList(
      [
        torch.nn.Parameter(
          torch.nn.init.xavier_normal_(torch.empty(in_features, self._low_rank))
        )
        for i in range(self._num_layers)
      ]
    )
    self.V_kernels: torch.nn.ParameterList = torch.nn.ParameterList(
      [
        torch.nn.Parameter(
          torch.nn.init.xavier_normal_(torch.empty(self._low_rank, in_features))
        )
        for i in range(self._num_layers)
      ]
    )
    self.bias: torch.nn.ParameterList = torch.nn.ParameterList(
      [
        torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(in_features)))
        for i in range(self._num_layers)
      ]
    )

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    """
    Args:
        input (torch.Tensor): tensor with shape [batch_size, in_features].

    Returns:
        torch.Tensor: tensor with shape [batch_size, in_features].
    """

    x_0 = input
    x_l = x_0

    for layer in range(self._num_layers):
      x_l_v = torch.nn.functional.linear(x_l, self.V_kernels[layer])
      x_l_w = torch.nn.functional.linear(x_l_v, self.W_kernels[layer])
      x_l = x_0 * (x_l_w + self.bias[layer]) + x_l  # (B, N)

    return x_l
