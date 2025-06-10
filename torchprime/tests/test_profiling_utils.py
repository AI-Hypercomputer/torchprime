"""Tests for profiling utility functions."""

from omegaconf import OmegaConf

from torchprime.utils.profiling import ensure_profile_end_step


def test_ensure_profile_end_step_sets_default():
  """ensure_profile_end_step sets profile_end_step when missing."""
  cfg = OmegaConf.create({"profile_start_step": 5, "profile_end_step": None})
  ensure_profile_end_step(cfg)
  assert cfg.profile_end_step == 15

