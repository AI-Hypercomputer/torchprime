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
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import transformers
from omegaconf import OmegaConf
from torch.utils.data import Dataset

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
  def __getitem__(self, idx):
    return {
      "input_ids": torch.ones(4),
      "attention_mask": torch.ones(4),
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
        "pure_modules": [],
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
        "max_grad_norm": None,
        "max_grad_value": None,
        "lr_scheduler": {"type": "constant", "warmup_steps": 0},
      },
      "run_name": None,
      "output_dir": "/tmp/test_output",
      "logging_steps": 1,
      "profile_start_step": -1,
      "profile_end_step": -1,
      "profile_dir": "/tmp/profile",
      "ici_mesh": {"data": 1, "fsdp": 1, "tensor": 1, "context": 1},
      "dcn_mesh": {},
    }
  )


def test_trainer_initialization(monkeypatch, dummy_config):
  """Test Trainer initialization and mesh logic."""
  # Arrange
  from torchprime.torch_xla_models.model_rewriting import sharding_initialization

  monkeypatch.setattr(
    sharding_initialization, "get_mesh", lambda *args, **kwargs: FakeMesh()
  )
  monkeypatch.setattr(
    sharding_initialization,
    "shard_torch_xla_model_from_config",
    lambda model, *args, **kwargs: model,
  )

  device = torch_xla.device()
  dataset = DummyDataset()

  # Act
  model = DummyModel().to(device)
  trainer = Trainer(model, dummy_config, dataset)

  # Assert
  assert isinstance(trainer.model, DummyModel)
  assert trainer.global_batch_size == 4


def test_trainer_optimizer(dummy_config):
  # Arrange
  device = torch_xla.device()
  model = DummyModel().to(device)

  # Act #1
  dummy_config.task.optimizer.type = "adafactor"
  opt = Trainer._create_optimizer(dummy_config, model.parameters())

  # Assert #1
  assert isinstance(opt, transformers.optimization.Adafactor)

  # Act #2
  dummy_config.task.optimizer.type = "adamw"
  dummy_config.task.optimizer.weight_decay = 1e-3
  opt = Trainer._create_optimizer(dummy_config, model.parameters())

  # Assert #2
  assert isinstance(opt, torch.optim.AdamW)
  assert opt.defaults["weight_decay"] == 1e-3

  # Assert #3
  with pytest.raises(ValueError, match=r"Supported optimizers are *"):
    # Act #3
    dummy_config.task.optimizer.type = "sgd"
    opt = Trainer._create_optimizer(dummy_config, model.parameters())


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

  trainer.train_loop()
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
  loss, grad_norm = trainer.train_step(batch)

  assert isinstance(loss, torch.Tensor)
  assert loss.ndim == 0  # scalar loss
  assert isinstance(grad_norm, torch.Tensor)
  assert grad_norm.ndim == 0  # scalar gradient norm


def test_trainer_clip_gradients_by_norm(monkeypatch, dummy_config):
  """Test correctness of gradient clipping by norm in a train step."""
  import torch_xla

  from torchprime.torch_xla_models.model_rewriting import sharding_initialization

  # Arrange
  monkeypatch.setattr(
    sharding_initialization, "get_mesh", lambda *args, **kwargs: FakeMesh()
  )
  monkeypatch.setattr(
    sharding_initialization,
    "shard_torch_xla_model_from_config",
    lambda model, *args, **kwargs: model,
  )

  class SumModel(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
      logits = self.linear(input_ids)
      loss = logits.mean()
      return logits, loss

  dummy_config.task.max_grad_norm = 1.0
  dummy_config.task.max_grad_value = None
  model = SumModel().to("xla")
  with torch.no_grad():
    model.linear.weight.fill_(1.0)
  dataset = DummyDataset()
  trainer = Trainer(model, dummy_config, dataset)
  torch_xla.sync()

  # Act
  batch = {k: v.unsqueeze(0).to("xla") for k, v in dataset[0].items()}
  loss, grad_norm = trainer.train_step(batch)

  # Assert
  # Loss should be exactly 4.0 since we are summing 4 inputs of 1.0.
  assert loss.item() == 4.0

  # ∂L/∂W = 1.0 for each weight in the linear layer
  # Expected gradient norm before clipping: sqrt(4 * 1^2) = 2.0
  assert pytest.approx(grad_norm.item(), rel=1e-5) == 2.0

  # Verify the actual gradients on the model
  # The original gradient for each weight would be 1.0
  # With clipping factor 0.5 (1.0/2.0), each gradient becomes 0.5
  if hasattr(model.linear.weight, "grad") and model.linear.weight.grad is not None:
    expected_clipped_grad = torch.full_like(model.linear.weight, 0.5)
    torch.testing.assert_close(
      model.linear.weight.grad, expected_clipped_grad, rtol=1e-5, atol=1e-5
    )

    # Also verify the gradient norm matches what we expect
    actual_grad_norm = torch.norm(model.linear.weight.grad)
    assert pytest.approx(actual_grad_norm.item(), rel=1e-5) == 1.0


def test_trainer_clip_gradients_by_value(monkeypatch, dummy_config):
  """Test correctness of gradient clipping by max absolute value in a train step."""
  import torch_xla

  from torchprime.torch_xla_models.model_rewriting import sharding_initialization

  # Arrange
  monkeypatch.setattr(
    sharding_initialization, "get_mesh", lambda *args, **kwargs: FakeMesh()
  )
  monkeypatch.setattr(
    sharding_initialization,
    "shard_torch_xla_model_from_config",
    lambda model, *args, **kwargs: model,
  )

  class SumModel(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
      logits = self.linear(input_ids)
      loss = logits.mean()
      return logits, loss

  dummy_config.task.max_grad_value = 0.5
  dummy_config.task.max_grad_norm = None
  model = SumModel().to("xla")
  with torch.no_grad():
    model.linear.weight.fill_(1.0)
  dataset = DummyDataset()
  trainer = Trainer(model, dummy_config, dataset)
  torch_xla.sync()

  # Act
  batch = {k: v.unsqueeze(0).to("xla") for k, v in dataset[0].items()}
  loss, grad_norm = trainer.train_step(batch)

  # Assert
  # Loss should be exactly 4.0 since we are summing 4 inputs of 1.0.
  assert loss.item() == 4.0

  # ∂L/∂W = 1.0 for each weight in the linear layer
  # Expected gradient norm before clipping: sqrt(4 * 1^2) = 2.0
  assert pytest.approx(grad_norm.item(), rel=1e-5) == 2.0

  # Verify the actual gradients on the model
  # The original gradient for each weight would be 1.0
  # With value clipping at 0.5, each gradient becomes 0.5
  if hasattr(model.linear.weight, "grad") and model.linear.weight.grad is not None:
    expected_clipped_grad = torch.full_like(model.linear.weight, 0.5)
    torch.testing.assert_close(
      model.linear.weight.grad, expected_clipped_grad, rtol=1e-5, atol=1e-5
    )

    # Verify all gradient values are within [-max_grad_value, max_grad_value]
    assert torch.all(model.linear.weight.grad <= dummy_config.task.max_grad_value)
    assert torch.all(model.linear.weight.grad >= -dummy_config.task.max_grad_value)


def test_profiler_trace(monkeypatch, dummy_config):
  """Verify profiler start_trace and stop_trace are invoked at configured steps."""
  from torchprime.torch_xla_models.model_rewriting import sharding_initialization

  monkeypatch.setattr(
    sharding_initialization, "get_mesh", lambda *args, **kwargs: FakeMesh()
  )
  monkeypatch.setattr(
    sharding_initialization,
    "shard_torch_xla_model_from_config",
    lambda model, *args, **kwargs: model,
  )

  calls = {"start": 0, "stop": 0}

  def fake_start(dir):
    calls["start"] += 1

  def fake_stop():
    calls["stop"] += 1

  monkeypatch.setattr(xp, "start_trace", fake_start)
  monkeypatch.setattr(xp, "stop_trace", fake_stop)
  monkeypatch.setattr(
    "torchprime.torch_xla_models.trainer.base_trainer.step_duration_from_latest_profile",
    lambda *_args, **_kwargs: 0.0,
  )

  dummy_config.profile_start_step = 0
  dummy_config.profile_end_step = 1
  dummy_config.task.max_steps = 6

  device = torch_xla.device()
  model = DummyModel().to(device)
  dataset = DummyDataset()
  trainer = Trainer(model, dummy_config, dataset)

  trainer.train_loop()

  assert dummy_config.profile_start_step == 0
  assert dummy_config.profile_end_step == 1
  assert calls["start"] == 1
  assert calls["stop"] == 1
