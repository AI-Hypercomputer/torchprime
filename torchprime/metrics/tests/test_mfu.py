import sys
from pathlib import Path

import pytest
import yaml

from torchprime.metrics.mfu import compute_mfu

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))


CONFIG_DIR = (
  Path(__file__).resolve().parents[2] / "torch_xla_models" / "configs" / "model"
)


def _load(name: str) -> dict:
  with open(CONFIG_DIR / name) as f:
    return yaml.safe_load(f)


def test_llama3_8b_mfu():
  cfg = _load("llama-3-8b.yaml")
  result = compute_mfu(
    cfg,
    batch_size=1024,
    sequence_length=4096,
    step_duration=2.801027417,
    tpu_name="foobar-v5p-512",
  )
  assert result.mfu == pytest.approx(0.6148650326, rel=0.01, abs=0.005)


def test_llama3_1_70b_mfu():
  cfg = _load("llama-3.1-70b.yaml")
  result = compute_mfu(
    cfg,
    batch_size=128,
    sequence_length=8192,
    step_duration=16.992,
    tpu_name="abc-v6e-128-stuff",
  )
  assert result.mfu == pytest.approx(0.2359197522, rel=0.01, abs=0.005)


def test_mixtral_8x7b_mfu():
  cfg = _load("mixtral-8x7b.yaml")
  result = compute_mfu(
    cfg,
    batch_size=1024,
    sequence_length=4096,
    step_duration=5.04,
    tpu_name="v6e-256",
  )
  assert result.mfu == pytest.approx(0.2822763546, rel=0.01, abs=0.005)


def test_deepseek_v3_mfu():
  cfg = _load("deepseek-v3.yaml")
  result = compute_mfu(
    cfg,
    batch_size=512,
    sequence_length=4096,
    step_duration=10.0,
    tpu_name="v5p-256",
  )
  assert result.mfu == pytest.approx(0.8939805997, rel=0.01, abs=0.005)
