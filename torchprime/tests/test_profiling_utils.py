"""Tests for profiling utility functions."""

from omegaconf import OmegaConf

from torchprime.utils.profiling import ensure_profile_end_step


def test_ensure_profile_end_step_sets_default():
  """ensure_profile_end_step sets profile_end_step when missing."""
  cfg = OmegaConf.create(
    {"profile_start_step": 5, "profile_end_step": None, "task": {"max_steps": 40}}
  )
  ensure_profile_end_step(cfg)
  assert cfg.profile_end_step == 25

  cfg = OmegaConf.create(
    {"profile_start_step": 5, "profile_end_step": None, "task": {"max_steps": 15}}
  )
  ensure_profile_end_step(cfg)
  assert cfg.profile_end_step == 10

  cfg = OmegaConf.create(
    {"profile_start_step": -1, "profile_end_step": None, "task": {"max_steps": 15}}
  )
  ensure_profile_end_step(cfg)
  assert cfg.profile_end_step == -1

  cfg = OmegaConf.create(
    {"profile_start_step": 5, "profile_end_step": 100, "task": {"max_steps": 15}}
  )
  ensure_profile_end_step(cfg)
  assert cfg.profile_end_step == 10

  cfg = OmegaConf.create(
    {"profile_start_step": 5, "profile_end_step": 20, "task": {"max_steps": 100}}
  )
  ensure_profile_end_step(cfg)
  assert cfg.profile_end_step == 20

  cfg = OmegaConf.create(
    {"profile_start_step": 5, "profile_end_step": None, "task": {"max_steps": 15}}
  )
  ensure_profile_end_step(cfg, num_profile_steps=3)
  assert cfg.profile_end_step == 8

  cfg = OmegaConf.create(
    {"profile_start_step": 5, "profile_end_step": 10, "task": {"max_steps": 8}}
  )
  ensure_profile_end_step(cfg)
  assert cfg.profile_end_step == 6
