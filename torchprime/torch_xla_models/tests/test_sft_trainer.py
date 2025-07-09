"""Tests for the :class:`SFTTrainer` class."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from torchprime.torch_xla_models.trainer.sft_trainer import SFTTrainer


class DummyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(4, 2)
    self.loaded = False
    self.saved = False

  def forward(self, input_ids=None, attention_mask=None, **kwargs):
    logits = self.linear(input_ids)
    loss = logits.mean()
    return logits, loss

  def from_pretrained(self, path):
    self.loaded = True

  def export(self, path):
    self.saved = True

  def _maybe_save_checkpoint(self, config):
    self.saved = True


class DummyDataset(Dataset):
  def __init__(self):
    self.device = xm.xla_device()

  def __getitem__(self, idx):
    return {
      "input_ids": torch.ones(4, device=self.device),
      "attention_mask": torch.ones(4, device=self.device),
    }

  def __len__(self):
    return 8


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
        "pure_modules": [],
        "remat": {
          "activation_checkpoint_layers": [],
          "optimization_barrier_layers": [],
          "scan_layers": None,
          "offload_tensors": [],
        },
        "sharding": {"type": "spmd"},
        "pretrained_model": "dummy",
      },
      "data": {"name": "dummy_dataset", "block_size": 4},
      "task": {
        "name": "sft",
        "global_batch_size": 4,
        "max_steps": 1,
        "max_grad_norm": None,
        "max_grad_value": None,
        "export_checkpoint_path": "dummy_export_path",
        "optimizer": {"type": "adafactor", "learning_rate": 1e-3},
        "lr_scheduler": {"type": "constant", "warmup_steps": 0},
      },
      "run_name": None,
      "output_dir": "/tmp/test_output",
      "logging_steps": 1,
      "profile_step": -1,
      "profile_dir": "/tmp/profile",
      "ici_mesh": {"data": 1, "fsdp": 1, "tensor": 1, "context": 1},
      "dcn_mesh": {},
    }
  )


def test_load_and_save(monkeypatch, dummy_config):
  from torchprime.torch_xla_models.model_rewriting import sharding_initialization

  # Patch mesh setup
  monkeypatch.setattr(
    sharding_initialization, "get_mesh", lambda *args, **kwargs: FakeMesh()
  )
  monkeypatch.setattr(
    sharding_initialization,
    "shard_torch_xla_model_from_config",
    lambda model, *args, **kwargs: model,
  )

  # Patch process index and count
  monkeypatch.setattr("torch_xla.runtime.process_index", lambda: 0)
  monkeypatch.setattr("torch_xla.runtime.process_count", lambda: 1)

  # Initialize
  device = xm.xla_device()
  model = DummyModel().to(device)
  dataset = DummyDataset()
  trainer = SFTTrainer(model, dummy_config, dataset)

  # from_pretrained should mark model as loaded
  assert model.loaded is True

  # Train (1 step), should trigger save
  trainer.train_loop()

  # Save should have occurred
  assert model.saved is True
