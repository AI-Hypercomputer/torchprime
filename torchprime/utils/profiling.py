"""Profiling utilities for trainer configuration."""

from omegaconf import DictConfig


def ensure_profile_end_step(
  config: DictConfig,
  num_profile_steps: int = 10,
) -> None:
  """Set ``profile_end_step`` based on ``profile_start_step`` if missing.

  Args:
    config: Trainer configuration object.
    num_profile_steps: Number of steps to trace when ``profile_end_step`` is ``None``.

  Returns:
    None. ``config`` is modified in place.
  """
  start = getattr(config, "profile_start_step", -1)
  end = getattr(config, "profile_end_step", None)
  num_profile_steps = getattr(config, "num_profile_steps", None) or num_profile_steps

  max_steps = config.task.max_steps
  if start < 0 or start >= max_steps:
    config.profile_start_step = -1
    config.profile_end_step = -1
    return config

  if end is None:
    end = start + num_profile_steps

  end = min(end, max_steps - 5)  # to prevent issue #260.

  config.profile_end_step = end
  return config
