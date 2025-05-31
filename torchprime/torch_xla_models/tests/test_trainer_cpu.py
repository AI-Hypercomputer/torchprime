"""Unit tests for the TPU Trainer class using PyTorch/XLA.

These tests validate the behavior of the Trainer class defined in
`torchprime.torch_xla_models.trainer.basic` by mocking TPU-specific and sharding logic
for CPU-based testing. It includes tests for:

- Trainer initialization logic and sharding setup.
- Single training loop execution with logging and step closures.
- XLA-compiled train step correctness.

Mocks are used to isolate TPU-specific components like mesh configuration,
device assignment, and compilation so tests can run on CPU.
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from torchprime.metrics.metrics import MetricsLogger
from torchprime.torch_xla_models.trainer.basic import Trainer


class DummyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(4, 2)

  def forward(self, input_ids=None, attention_mask=None, **kwargs):
    logits = self.linear(input_ids)
    loss = logits.mean()
    return logits, loss


class DummyDataset(Dataset):
  def __getitem__(self, idx):
    return {"input_ids": torch.ones(4), "attention_mask": torch.ones(4)}

  def __len__(self):
    return 100


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
      "global_batch_size": 4,
      "max_steps": 2,
      "output_dir": "/tmp/test_output",
      "logging_steps": 1,
      "profile_step": -1,
      "profile_dir": "/tmp/profile",
      "profile_duration": 5,
      "optimizer": {"type": "adafactor", "learning_rate": 1e-3},
      "lr_scheduler": {"type": "constant", "warmup_steps": 0},
      "block_size": 4,
      "model": {
        "remat": {
          "activation_checkpoint_layers": [],
          "optimization_barrier_layers": [],
          "scan_layers": None,
          "offload_tensors": [],
        },
        "sharding": {"type": "spmd"},
      },
      "ici_mesh": {
        "data": 1,
        "fsdp": 1,
        "tensor": 1,
      },
      "dcn_mesh": {},
    }
  )


@patch(
  "torchprime.torch_xla_models.sharding.initialization.get_mesh",
  return_value=FakeMesh(),
)
@patch(
  "torchprime.torch_xla_models.sharding.initialization.shard_torch_xla_model_from_config",
  side_effect=lambda model, *args, **kwargs: model,
)
@patch("torchprime.torch_xla_models.trainer.basic.xm.xla_device", return_value="cpu")
@patch("torchprime.torch_xla_models.trainer.basic.torch_xla.sync")
def test_trainer_init(
  mock_sync, mock_device, mock_shard_model, mock_get_mesh, dummy_config
):
  model = DummyModel()
  dataset = DummyDataset()
  trainer = Trainer(model, dummy_config, dataset)
  assert isinstance(trainer.model, DummyModel)
  assert trainer.global_batch_size == 4
  assert trainer.device == "cpu"


@patch(
  "torchprime.torch_xla_models.sharding.initialization.get_mesh",
  return_value=FakeMesh(),
)
@patch(
  "torchprime.torch_xla_models.sharding.initialization.shard_torch_xla_model_from_config",
  side_effect=lambda model, *args, **kwargs: model,
)
@patch("torchprime.torch_xla_models.trainer.basic.xm.xla_device", return_value="cpu")
@patch("torchprime.torch_xla_models.trainer.basic.torch_xla.sync")
@patch("torchprime.torch_xla_models.trainer.basic.xm.add_step_closure")
@patch("torchprime.torch_xla_models.trainer.basic.xm.wait_device_ops")
@patch("torchprime.torch_xla_models.trainer.basic.Trainer._get_train_dataloader")
@patch("torchprime.torch_xla_models.trainer.basic.Trainer.train_step")
def test_train_loop(
  mock_train_step,
  mock_get_loader,
  mock_wait,
  mock_closure,
  mock_sync,
  mock_device,
  mock_shard_model,
  mock_get_mesh,
  dummy_config,
):
  model = DummyModel()
  dataset = DummyDataset()
  mock_get_loader.return_value = iter([dataset[0], dataset[0]])
  mock_train_step.return_value = torch.tensor(0.1)

  trainer = Trainer(model, dummy_config, dataset)
  trainer.train_loop(metrics_logger=MetricsLogger())
  assert mock_train_step.call_count == dummy_config.max_steps


@patch(
  "torchprime.torch_xla_models.trainer.basic.torch_xla.compile",
  lambda **kwargs: lambda fn: fn,
)
@patch(
  "torchprime.torch_xla_models.sharding.initialization.get_mesh",
  return_value=FakeMesh(),
)
@patch(
  "torchprime.torch_xla_models.sharding.initialization.shard_torch_xla_model_from_config",
  side_effect=lambda model, *args, **kwargs: model,
)
@patch("torchprime.torch_xla_models.trainer.basic.xm.xla_device", return_value="cpu")
@patch("torchprime.torch_xla_models.trainer.basic.torch_xla.sync")
def test_train_step(
  mock_sync, mock_device, mock_shard_model, mock_get_mesh, dummy_config
):
  model = DummyModel()
  dataset = DummyDataset()
  batch = {k: v.unsqueeze(0) for k, v in dataset[0].items()}
  trainer = Trainer(model, dummy_config, dataset)
  loss = trainer.train_step(batch)
  assert isinstance(loss, torch.Tensor)
