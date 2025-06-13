import unittest

import torch

import torchprime.utils.parallelism_utils as p_utils


class TestParallelism_utils(unittest.TestCase):
  def test_reorder_sequence_forward(self):
    input = torch.arange(8)
    res = p_utils.reorder_sequence(
      tensor=input, cp_size=2, seq_dim=0, to_contiguous=False
    )
    assert (res == torch.tensor([0, 1, 6, 7, 2, 3, 4, 5])).all()

  def test_reorder_sequence_backward(self):
    input = torch.tensor([0, 1, 6, 7, 2, 3, 4, 5])
    res = p_utils.reorder_sequence(
      tensor=input, cp_size=2, seq_dim=0, to_contiguous=True
    )
    assert (res == torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])).all()

  def test_reorder_sequence_forward_2d(self):
    input = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]])
    res = p_utils.reorder_sequence(
      tensor=input, cp_size=2, seq_dim=1, to_contiguous=False
    )
    assert (
      res == torch.tensor([[0, 1, 6, 7, 2, 3, 4, 5], [0, 1, 6, 7, 2, 3, 4, 5]])
    ).all()

  def test_loadBalancedCasualMask(self):
    mask_shape = (8, 8)
    lbMask = p_utils.LoadBalancedCausalMask(shape=mask_shape, cp_size=4)
    assert (lbMask.q_sequence == torch.tensor([0, 7, 1, 6, 2, 5, 3, 4])).all()
