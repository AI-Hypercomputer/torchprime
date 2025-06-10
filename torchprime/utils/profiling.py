"""Profiling utilities for trainer configuration."""

from omegaconf import DictConfig


def ensure_profile_end_step(config: DictConfig, default_span: int = 7) -> None:
  """Set ``profile_end_step`` based on ``profile_start_step`` if missing.

  Args:
    config: Trainer configuration object.
    default_span: Number of steps to trace when ``profile_end_step`` is ``None``.

  Returns:
    None. ``config`` is modified in place.
  """
  start = getattr(config, "profile_start_step", -1)
  end = getattr(config, "profile_end_step", None)
  if start >= 0 and end is None:
    config.profile_end_step = start + default_span
