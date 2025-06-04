"""Unit tests for the TPU Trainer class using PyTorch/XLA.

These tests validate the behavior of the Trainer class defined in
`torchprime.torch_xla_models.trainer.base_trainer` using dummy model/data and
a fake mesh, while running real training logic on XLA device.

Includes:
- Trainer initialization logic and mesh setup.
- Full training loop execution with train_step call tracking.
- Single train step correctness.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from torchprime.metrics.metrics import MetricsLogger
from torchprime.torch_xla_models.trainer.base_trainer import Trainer


class DummyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(4, 2)

  def forward(self, input_ids=None, attention_mask=None, **kwargs):
    logits = self.linear(input_ids)
    loss = logits.mean()
    return logits, loss


class DummyDataset(Dataset):
  def __init__(self):
    self.device = xm.xla_device()

  def __getitem__(self, idx):
    return {
      "input_ids": torch.ones(4, device=self.device),
      "attention_mask": torch.ones(4, device=self.device),
    }

  def __len__(self):
    return 16


# TODO: Replace FakeMesh with a real mesh implementation when available.
class FakeMesh:
  def __init__(self):
    self.device_ids = [0]
    self.axis_names = ("data", "fsdp")
    self.mesh_shape = (1, 1)

  def shape(self):
    return {"data": 1, "fsdp": 1}

  def get_axis_name_idx(self, axis_name):
    return self.axis_names.index(axis_name)

  def get_logical_mesh(self):
    return np.array(self.device_ids).reshape(self.mesh_shape)


@pytest.fixture
def dummy_config():
  return OmegaConf.create(
    {
      "model": {
        "remat": {
          "activation_checkpoint_layers": [],
          "optimization_barrier_layers": [],
          "scan_layers": None,
          "offload_tensors": [],
        },
        "sharding": {"type": "spmd"},
      },
      "data": {"name": "dummy_dataset", "block_size": 4},
      "task": {
        "name": "dummy_task",
        "global_batch_size": 4,
        "max_steps": 2,
        "optimizer": {"type": "adafactor", "learning_rate": 1e-3},
        "lr_scheduler": {"type": "constant", "warmup_steps": 0},
      },
      "run_name": None,
      "output_dir": "/tmp/test_output",
      "logging_steps": 1,
      "profile_step": -1,
      "profile_dir": "/tmp/profile",
      "profile_duration": 5,
      "ici_mesh": {"data": 1, "fsdp": 1, "tensor": 1},
      "dcn_mesh": {},
    }
  )


def test_trainer_initialization(monkeypatch, dummy_config):
  """Test Trainer initialization and mesh logic."""
  from torchprime.torch_xla_models.model_rewriting import sharding_initialization

  monkeypatch.setattr(
    sharding_initialization, "get_mesh", lambda *args, **kwargs: FakeMesh()
  )
  monkeypatch.setattr(
    sharding_initialization,
    "shard_torch_xla_model_from_config",
    lambda model, *args, **kwargs: model,
  )

  device = xm.xla_device()
  model = DummyModel().to(device)
  dataset = DummyDataset()
  trainer = Trainer(model, dummy_config, dataset)

  assert isinstance(trainer.model, DummyModel)
  assert trainer.global_batch_size == 4


def test_trainer_train_loop(monkeypatch, dummy_config):
  """Test full training loop execution and step count via wrapper."""
  from torchprime.torch_xla_models.model_rewriting import sharding_initialization

  monkeypatch.setattr(
    sharding_initialization, "get_mesh", lambda *args, **kwargs: FakeMesh()
  )
  monkeypatch.setattr(
    sharding_initialization,
    "shard_torch_xla_model_from_config",
    lambda model, *args, **kwargs: model,
  )

  device = xm.xla_device()
  model = DummyModel().to(device)
  dataset = DummyDataset()
  trainer = Trainer(model, dummy_config, dataset)

  # Count how many times train_step is called
  call_counter = {"steps": 0}
  original_train_step = Trainer.train_step

  def counting_train_step(self, batch):
    call_counter["steps"] += 1
    return original_train_step(self, batch)

  monkeypatch.setattr(Trainer, "train_step", counting_train_step)

  trainer.train_loop(metrics_logger=MetricsLogger())
  assert call_counter["steps"] == dummy_config.task.max_steps


def test_trainer_train_step(monkeypatch, dummy_config):
  """Test correctness of one training step."""
  from torchprime.torch_xla_models.model_rewriting import sharding_initialization

  monkeypatch.setattr(
    sharding_initialization, "get_mesh", lambda *args, **kwargs: FakeMesh()
  )
  monkeypatch.setattr(
    sharding_initialization,
    "shard_torch_xla_model_from_config",
    lambda model, *args, **kwargs: model,
  )

  device = xm.xla_device()
  model = DummyModel().to(device)
  dataset = DummyDataset()
  trainer = Trainer(model, dummy_config, dataset)

  batch = {k: v.unsqueeze(0).to(device) for k, v in dataset[0].items()}
  loss = trainer.train_step(batch)

  assert isinstance(loss, torch.Tensor)
  assert loss.ndim == 0  # scalar loss
