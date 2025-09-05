from pathlib import Path

import pytest
import yaml

from torchprime.metrics.mfu import compute_mfu

ABS_CONFIG_PATH = (Path(__file__).parent / "../../torch_xla_models/configs").resolve()


def _load_model_config(config_filename: str) -> dict:
  """Loads a specific model config as a standard Python dictionary."""

  model_config_path = ABS_CONFIG_PATH / "model" / config_filename

  with open(model_config_path) as f:
    return yaml.safe_load(f)


def test_llama3_8b_mfu():
  cfg = _load_model_config("llama-3-8b.yaml")
  result = compute_mfu(
    cfg,
    batch_size=1024,
    sequence_length=4096,
    step_duration=2.801027417,
    tpu_name="foobar-v5p-512",
  )
  assert result.mfu == pytest.approx(0.6148650326, rel=0.01, abs=0.005)


def test_llama3_1_70b_mfu():
  cfg = _load_model_config("llama-3.1-70b.yaml")
  result = compute_mfu(
    cfg,
    batch_size=128,
    sequence_length=8192,
    step_duration=16.992,
    tpu_name="abc-v6e-128-stuff",
  )
  assert result.mfu == pytest.approx(0.2359197522, rel=0.01, abs=0.005)


def test_mixtral_8x7b_mfu():
  cfg = _load_model_config("mixtral-8x7b.yaml")
  result = compute_mfu(
    cfg,
    batch_size=1024,
    sequence_length=4096,
    step_duration=5.04,
    tpu_name="v6e-256",
  )
  assert result.mfu == pytest.approx(0.2822763546, rel=0.01, abs=0.005)


def test_deepseek_v3_mfu():
  cfg = _load_model_config("deepseek-v3.yaml")
  result = compute_mfu(
    cfg,
    batch_size=512,
    sequence_length=4096,
    step_duration=10.0,
    tpu_name="v5p-256",
  )
  assert result.mfu == pytest.approx(0.8939805997, rel=0.01, abs=0.005)
