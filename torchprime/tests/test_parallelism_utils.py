import unittest

import numpy as np
import torch
import torch_xla

import torchprime.utils.parallelism_utils as p_utils


class TestParallelism_utils(unittest.TestCase):
  def test_reorder_sequence_forward(self):
    input = torch.arange(8).to(torch_xla.device())
    res = p_utils.reorder_sequence(
      tensor=input, cp_size=2, seq_dim=0, to_contiguous=False
    )
    output_expected = torch.tensor([0, 1, 6, 7, 2, 3, 4, 5], device=torch_xla.device())
    assert (res == output_expected).all()

  def test_reorder_sequence_backward(self):
    input = torch.tensor([0, 1, 6, 7, 2, 3, 4, 5]).to(torch_xla.device())
    res = p_utils.reorder_sequence(
      tensor=input, cp_size=2, seq_dim=0, to_contiguous=True
    )
    output_expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=torch_xla.device())
    assert (res == output_expected).all()

  def test_reorder_sequence_forward_2d(self):
    input = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]).to(
      torch_xla.device()
    )
    res = p_utils.reorder_sequence(
      tensor=input, cp_size=2, seq_dim=1, to_contiguous=False
    )
    output_expected = torch.tensor(
      [[0, 1, 6, 7, 2, 3, 4, 5], [0, 1, 6, 7, 2, 3, 4, 5]], device=torch_xla.device()
    )
    assert (res == output_expected).all()

  def test_load_balanced_casual_mask(self):
    mask_shape = (8, 8)
    lbMask = p_utils.LoadBalancedCausalMask(shape=mask_shape, cp_size=4)
    assert (lbMask.q_sequence == torch.tensor([0, 7, 1, 6, 2, 5, 3, 4])).all()

  def test_large_tensor(self):
    input = torch.randint(128256, ((8, 256)), device=torch_xla.device())
    input_reorder = p_utils.reorder_sequence(
      tensor=input,
      cp_size=2,
      seq_dim=1,
      to_contiguous=False,
    )
    input_unpermuted = p_utils.reorder_sequence(
      tensor=input_reorder,
      cp_size=2,
      seq_dim=1,
      to_contiguous=True,
    )
    torch.testing.assert_close(
      input.cpu(),
      input_unpermuted.cpu(),
    )

  def test_reorder_mask(self):
    input = np.arange(8)
    res = p_utils.reorder_mask(tensor=input, cp_size=2, seq_dim=0)
    output_expected = np.array([0, 1, 6, 7, 2, 3, 4, 5])
    assert (res == output_expected).all()

    input = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]])
    res = p_utils.reorder_mask(tensor=input, cp_size=2, seq_dim=1)
    output_expected = np.array([[0, 1, 6, 7, 2, 3, 4, 5], [0, 1, 6, 7, 2, 3, 4, 5]])
    assert (res == output_expected).all()
