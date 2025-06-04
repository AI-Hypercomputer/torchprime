"""Base trainer module for TPU-based model training using PyTorch/XLA.

This script provides a `Trainer` class that sets up model sharding, activation checkpointing,
optimization, and the training loop with XLA-specific configurations. It is designed to work with
distributed TPU training and includes utilities for metrics logging and MFU computation.

Typical usage example:

  trainer = Trainer(model, config, train_dataset)
  trainer.train_loop(metrics_logger)
"""

import logging
import math
import os
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
  default_data_collator,
  get_scheduler,
)
from transformers.optimization import Adafactor

from torchprime.metrics.mfu import compute_mfu
from torchprime.metrics.step_duration import step_duration_from_latest_profile
from torchprime.torch_xla_models.model_rewriting.rematerialization_utils import (
  add_activation_checkpointing_and_scan,
  add_optimization_barriers,
)
from torchprime.torch_xla_models.model_rewriting.sharding_initialization import (
  setup_sharding_and_mesh,
)
from torchprime.torch_xla_models.topology import get_num_slices

logger = logging.getLogger(__name__)


def get_model_dtype(module: nn.Module) -> torch.dtype:
  dtypes = {param.dtype for param in module.parameters()}
  if len(dtypes) != 1:
    raise ValueError(f"Inconsistent dtypes found: {dtypes}")
  return dtypes.pop()


class Trainer:
  """Trainer class for TPU-accelerated model training using PyTorch/XLA.

  This class encapsulates model preparation, optimizer configuration, data loading,
  and the training loop. It is designed to handle distributed training across TPU cores,
  enabling features like SPMD sharding, activation checkpointing, and profiling.

  Args:
    model: The model to train.
    config: Configuration object containing training hyperparameters and setup.
    train_dataset: Dataset used for training.
  """

  minibatch: bool

  def __init__(
    self,
    model: nn.Module,
    config: DictConfig,
    train_dataset: Dataset | IterableDataset | None,
  ):
    self.config = config
    self.device = xm.xla_device()
    self.global_batch_size = self.config.global_batch_size
    self.train_dataset = train_dataset

    # Initialize tensorboard metrics writer
    self._initialize_tensorboard_writer()

    # Sharding setup
    model, self.input_sharding_spec, self.minibatch = setup_sharding_and_mesh(
      model, config
    )

    # Model transformations
    model = add_activation_checkpointing_and_scan(model, config)
    model = add_optimization_barriers(model, config)
    self.model = model

    assert self.config.optimizer.type == "adafactor", (
      "Currently only Adafactor optimizer is supported"
    )

    # Set up optimizers
    self.optimizer = Adafactor(
      params=model.parameters(),
      lr=self.config.optimizer.learning_rate,
      relative_step=False,
      scale_parameter=False,
    )

    # TODO: this OOMs the TPU.
    # self._prime_optimizer()

    self.lr_scheduler = get_scheduler(
      name=self.config.lr_scheduler.type,
      optimizer=self.optimizer,
      num_warmup_steps=self.config.lr_scheduler.warmup_steps,
      num_training_steps=self.config.max_steps,
    )

    # Execute all initialization work queued so far before starting training.
    torch_xla.sync()

  def __del__(self):
    # Close TensorBoard writer on destruction.
    self.summary_writer.close()

  def _initialize_tensorboard_writer(self):
    run_name = self.config.run_name
    if run_name is None:
      run_name = datetime.now().strftime("%b%d_%H-%M-%S")
    tensorboard_dir = Path(self.config.output_dir) / "runs" / run_name
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"TensorBoard logging to: {tensorboard_dir}")
    self.summary_writer = SummaryWriter(log_dir=str(tensorboard_dir))

  def _prime_optimizer(self) -> None:
    for group in self.optimizer.param_groups:
      for p in group["params"]:
        p.grad = torch.zeros_like(p)
        p.grad.requires_grad_(False)
    self.optimizer.step()
    torch_xla.sync()

  def _get_train_dataloader(self) -> pl.MpDeviceLoader:
    if self.train_dataset is None:
      raise ValueError("Trainer: training requires a train_dataset.")

    num_replicas = xr.process_count()
    logger.info("Num replicas: %d", num_replicas)

    if self.minibatch:
      sampler = torch.utils.data.DistributedSampler(
        self.train_dataset,
        num_replicas=num_replicas,
        rank=xr.process_index(),
      )
    else:
      # Without minibatch, every process loads the global batch the same way.
      sampler = torch.utils.data.DistributedSampler(
        self.train_dataset,
        num_replicas=1,
        rank=0,
      )

    assert self.global_batch_size is not None
    if self.minibatch:
      # Each process loads the per-host batch size.
      batch_size = self.global_batch_size // num_replicas
    else:
      # Each process will load the global batch, then discard the unneeded parts.
      batch_size = self.global_batch_size

    dataloader = DataLoader(
      self.train_dataset,
      # Data collator will default to DataCollatorWithPadding, so we change it.
      collate_fn=default_data_collator,
      batch_size=batch_size,
      sampler=sampler,
      drop_last=True,
    )
    loader = pl.MpDeviceLoader(
      dataloader, self.device, input_sharding=self.input_sharding_spec
    )
    return loader

  def _log_to_tensorboard(
    self, epoch: float, step: int, loss: float, learning_rate: float
  ):
    """Log metrics to TensorBoard."""
    self.summary_writer.add_scalar("train/epoch", epoch, step)
    self.summary_writer.add_scalar("train/loss", loss, step)
    self.summary_writer.add_scalar("train/learning_rate", learning_rate, step)
    self.summary_writer.flush()

  def train_loop(self, metrics_logger) -> None:
    self.model.train()
    self.model.zero_grad()

    # For now we assume that we will never train for more than one epoch
    max_step = self.config.max_steps
    train_loader = self._get_train_dataloader()
    train_iterator = iter(train_loader)

    logger.info("Starting training")
    logger.info("    Max step: %d", max_step)
    logger.info("    Global batch size: %d", self.global_batch_size)

    epoch = 0
    for step in range(max_step):
      try:
        batch = next(train_iterator)
      except StopIteration:
        logger.warning("DataLoader exhausted at step %d, reset iterator", step)
        epoch += 1
        train_iterator = iter(train_loader)
        batch = next(train_iterator)

      trace_start_time = timer()
      loss = self.train_step(batch)
      trace_end_time = timer()

      if step % self.config.logging_steps == 0:

        def step_closure(epoch, step, loss, trace_start_time, trace_end_time, lr):
          loss = loss.detach().item()
          logger.info(
            "Epoch: %d, step: %d, loss: %.4f, lr: %.2e, trace time: %.2f ms",
            epoch,
            step,
            loss,
            lr,
            (trace_end_time - trace_start_time) * 1000,
          )
          self._log_to_tensorboard(epoch, step, loss, lr)
          if math.isnan(loss):
            raise ValueError(f"Loss is NaN at step {step}")

        xm.add_step_closure(
          step_closure,
          args=(
            epoch,
            step,
            loss,
            trace_start_time,
            trace_end_time,
            self.lr_scheduler.get_last_lr()[0],
          ),
          run_async=True,
        )

      # Capture profile at the prefer step
      if step == self.config.profile_step:
        # Wait until device execution catches up to tracing before triggering the profile. This will
        # interrupt training slightly on the hosts which are capturing, but by waiting after tracing
        # for the step, the interruption will be minimal.
        xm.wait_device_ops()
        xp.trace_detached(
          "127.0.0.1:9012",
          self.config.profile_dir,
          self.config.profile_duration,
        )

    xm.wait_device_ops()
    logger.info("Finished training run")

    if self.config.profile_step >= 0:
      # Analyze the step duration from the latest profile
      step_duration = step_duration_from_latest_profile(self.config.profile_dir)
      metrics_logger.log_step_execution_time(step_duration)

      tpu_name = os.environ.get("TORCHPRIME_TPU_TYPE", None)
      if tpu_name:
        # Compute MFU
        mfu = compute_mfu(
          config=self.config.model,
          batch_size=self.config.global_batch_size,
          step_duration=step_duration,
          tpu_name=tpu_name,
          num_slices=get_num_slices(),
          sequence_length=self.config.block_size,
          torch_dtype=self.config.torch_dtype,
        )
        metrics_logger.log_mfu(mfu.mfu)

        # Compute tokens per seconds
        tokens_per_second = (
          self.config.block_size * self.config.global_batch_size // step_duration
        )
        metrics_logger.log_tokens_per_second(tokens_per_second)

        # Log number of steps
        metrics_logger.log_num_steps(self.config.max_steps)

    # Print and save metrics
    metrics = metrics_logger.finalize()
    logger.info("***** train metrics *****\n%s", metrics)
    metrics.save(Path(self.config.output_dir) / "train_metrics.json")

    # Save the hydra config
    config_save_path = Path(self.config.output_dir) / "train_config.json"
    OmegaConf.save(config=self.config, f=config_save_path)

  @torch_xla.compile(full_graph=True)
  def train_step(self, batch: dict) -> torch.Tensor:
    _logits, loss = self.model(**batch)
    loss.backward()
    self.optimizer.step()
    self.lr_scheduler.step()
    self.model.zero_grad()
    return loss
