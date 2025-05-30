import importlib
import logging
import math
import os
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from timeit import default_timer as timer

import datasets
import hydra
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import transformers
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_xla._internal.jax_workarounds import jax_env_context
from torch_xla.distributed.fsdp import checkpoint_module
from torch_xla.distributed.spmd.xla_sharding import apply_xla_patch_to_nn_linear
from transformers import (
  AutoTokenizer,
  default_data_collator,
  get_scheduler,
  set_seed,
)
from transformers.optimization import Adafactor
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.utils import check_min_version

from torchprime.data.dataset import make_huggingface_dataset
from torchprime.layers.sequential import HomogeneousSequential
from torchprime.metrics.metrics import MetricsLogger
from torchprime.metrics.mfu import compute_mfu
from torchprime.metrics.step_duration import step_duration_from_latest_profile
from torchprime.sharding.shard_model import (
  shard_torch_xla_model_from_config,
  wrap_module,
)
from torchprime.torch_xla_models import offloading, remat_all, scan_layers
from torchprime.torch_xla_models.topology import (
  get_mesh,
  get_num_slices,
  is_1d_sharding,
)
from torchprime.utils.retry import retry

check_min_version("4.39.3")
logger = logging.getLogger(__name__)

xr.use_spmd()
assert xr.is_spmd() is True


class Trainer:
  """The trainer."""

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

    # Set up SPMD mesh and shard the model
    mesh = get_mesh(self.config)
    xs.set_global_mesh(mesh)
    logger.info(f"Logical mesh shape: {mesh.shape()}")
    logger.info(f"Logical mesh device assignments: {mesh.device_ids}")

    # TODO(https://github.com/pytorch/xla/issues/8696): Minibatch only works in 1D sharding.
    minibatch = is_1d_sharding(tuple(config.ici_mesh.values()))
    self.minibatch = minibatch
    logger.info(f"Minibatch dataloading: {minibatch}")

    # TODO(https://github.com/AI-Hypercomputer/torchprime/issues/66): Test this for multislice
    self.input_sharding_spec = xs.ShardingSpec(
      mesh, (("data", "fsdp"), None), minibatch=minibatch
    )

    # Recursively replace `nn.Linear` layers with einsum operations in the model.
    # Without this patch, an `nn.Linear` module will flatten non-contracting dimensions
    # (e.g. batch and sequence), thus destroying the sharding constraints on those dimensions.
    model = apply_xla_patch_to_nn_linear(model)

    # Annotate model weights and activations with sharding constraints to distribute
    # the training across devices following the SPMD paradigm.
    sharding_config = OmegaConf.to_container(self.config.model.sharding, resolve=True)
    assert isinstance(sharding_config, dict), (
      f"Sharding config {sharding_config} must be a dict"
    )
    model = shard_torch_xla_model_from_config(model, config=sharding_config)

    # Rematerialize forward computation during the backward pass if requested.
    model = self._add_checkpoint_offload_scan_model(model)
    model = self._add_optimization_barrier_model(model)
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

  def _prime_optimizer(self):
    for group in self.optimizer.param_groups:
      for p in group["params"]:
        p.grad = torch.zeros_like(p)
        p.grad.requires_grad_(False)
    self.optimizer.step()
    torch_xla.sync()

  def _get_train_dataloader(self):
    if self.train_dataset is None:
      raise ValueError("Trainer: training requires a train_dataset.")

    num_replicas = xr.process_count()
    logger.info(f"Num replicas: {num_replicas}")
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

  def _add_checkpoint_offload_scan_model(self, model: nn.Module):
    remat_classes = self._get_classes_by_names(
      model, self.config.model.remat.get("activation_checkpoint_layers", [])
    )
    layers_to_scan = self.config.model.remat.get("scan_layers", None)
    offload_tensors = self.config.model.remat.get("offload_tensors", [])

    # Checking preconditions and logging.
    if remat_classes:
      logger.info(f"Enabling activation checkpointing on {remat_classes}")
    if layers_to_scan:
      assert isinstance(layers_to_scan, str)
      logger.info(f"Compiling module `{layers_to_scan}` with scan")
    if len(offload_tensors):
      logger.info(f"Will offload these tensors to host RAM: {offload_tensors}")
      if layers_to_scan is None:
        raise NotImplementedError("Host offloading requires scan")
      if len(remat_classes) != 1:
        raise NotImplementedError(
          "Host offloading requires checkpointing exactly one layer"
        )

    def maybe_checkpoint(mod, _name):
      if isinstance(mod, tuple(remat_classes)):
        return checkpoint_module(mod)
      return mod

    if layers_to_scan is None:
      # Implement activation checkpointing without scan by wrapping modules.
      if not remat_classes:
        return model
      return wrap_module(model, maybe_checkpoint)

    if not remat_classes:
      # Scan without activation checkpointing.
      return scan_layers.compile(model, layers_to_scan)

    # Implement activation checkpointing and host offloading under scan via
    # a graph partitioner instead of `checkpoint_module`.
    seq = model.get_submodule(layers_to_scan)
    assert isinstance(seq, HomogeneousSequential)
    if len(remat_classes) != 1 or list(remat_classes)[0] != seq.repeated_layer:
      raise NotImplementedError(
        f"When compiling decoder layers with scan and \
          activation checkpointing is also requested, we only support \
          checkpointing {seq.repeated_layer} i.e. the layer being scanned."
      )
    if not len(offload_tensors):
      partition_fn = remat_all.remat_all_partition_fn
    else:
      partition_fn = partial(
        offloading.remat_all_and_offload_these_inputs,
        names_to_offload=offload_tensors,
      )
    return scan_layers.compile(model, layers_to_scan, partition_fn=partition_fn)

  def _add_optimization_barrier_model(self, model: nn.Module):
    classes = self._get_classes_by_names(
      model, self.config.model.remat.get("optimization_barrier_layers", [])
    )
    if not classes:
      return model

    logger.info(f"Adding backward optimization barriers to {classes}")

    def maybe_add_barrier(mod, _name):
      if isinstance(mod, tuple(classes)):
        # Register a backward hook to place optimization barrier to prevent
        # gigantic fusions on syncing the gradients.
        xs.apply_backward_optimization_barrier(mod)
        return mod
      return mod

    return wrap_module(model, maybe_add_barrier)

  def _get_classes_by_names(self, model, activation_checkpoint_layers: list[str]):
    classes_to_checkpoint = set()
    for layer_class in activation_checkpoint_layers:
      cls = get_module_class_from_name(model, layer_class)
      if cls is None:
        raise Exception(
          f"Could not find the transformer layer class {layer_class} in the model."
        )
      else:
        classes_to_checkpoint.add(cls)
    return tuple(classes_to_checkpoint)

  def train_loop(self, metrics_logger):
    self.model.train()
    self.model.zero_grad()

    # For now we assume that we wil never train for mor than one epoch
    max_step = self.config.max_steps
    train_loader = self._get_train_dataloader()
    train_iterator = iter(train_loader)

    logger.info("Starting training")
    logger.info(f"    Max step: {max_step}")
    logger.info(f"    Global batch size: {self.global_batch_size}")

    epoch = 0
    for step in range(max_step):
      try:
        batch = next(train_iterator)
      except StopIteration:
        logger.warning(f"DataLoader exhausted at step {step}, reset iterator")
        epoch += 1
        train_iterator = iter(train_loader)
        batch = next(train_iterator)

      trace_start_time = timer()
      loss = self.train_step(batch)
      trace_end_time = timer()

      if step % self.config.logging_steps == 0:

        def step_closure(epoch, step, loss, trace_start_time, trace_end_time):
          loss = loss.detach().item()
          logger.info(
            f"Epoch: {epoch}, step: {step}, loss: {loss:0.4f}, "
            f"trace time: {(trace_end_time - trace_start_time) * 1000:0.2f} ms"
          )
          if math.isnan(loss):
            raise ValueError(f"Loss is NaN at step {step}")

        xm.add_step_closure(
          step_closure,
          args=(epoch, step, loss, trace_start_time, trace_end_time),
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
  def train_step(self, batch):
    _logits, loss = self.model(**batch)
    loss.backward()
    self.optimizer.step()
    self.lr_scheduler.step()
    self.model.zero_grad()
    return loss


def initialize_model_class(model_config):
  """Import and initialize model_class specified by the config."""
  full_model_class_string = model_config.model_class
  module_name, model_class_name = full_model_class_string.rsplit(".", 1)
  try:
    module = importlib.import_module(module_name)
  except ModuleNotFoundError as e:
    print(f"Error importing relative module: {e}")
    sys.exit(1)
  if hasattr(module, model_class_name):
    model_class = getattr(module, model_class_name)
  else:
    print(f"Error: Function '{model_class_name}' not found in module '{module_name}'")
    sys.exit(1)
  model = model_class(model_config)
  return model


@contextmanager
def set_default_dtype(dtype):
  # Get the current default dtype
  previous_dtype = torch.get_default_dtype()
  # Set the new default dtype
  torch.set_default_dtype(dtype)
  try:
    yield
  finally:
    # Revert to the original default dtype
    torch.set_default_dtype(previous_dtype)


def get_model_dtype(module):
  dtypes = {param.dtype for param in module.parameters()}
  if len(dtypes) != 1:
    raise ValueError(f"Inconsistent dtypes found: {dtypes}")
  return dtypes.pop()


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: DictConfig):
  metrics_logger = (
    MetricsLogger()
  )  # Call metricslogger in the beginning to get correct start time.
  print(OmegaConf.to_yaml(config))  # Print the config for debugging
  log_level = logging.INFO
  logger.setLevel(log_level)
  logger.setLevel(log_level)
  datasets.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

  set_seed(config.seed)
  torch_xla.manual_seed(config.seed)
  server = xp.start_server(9012)
  logger.info(f"Profiling server started: {str(server)}")

  # TODO(https://github.com/AI-Hypercomputer/torchprime/issues/14): Add tokenizers to torchprime.
  tokenizer_name = config.model.tokenizer_name
  tokenizer = retry(lambda: AutoTokenizer.from_pretrained(tokenizer_name))

  assert config.torch_dtype == "bfloat16", "Currently only bfloat16 is supported"
  model_dtype = getattr(torch, config.torch_dtype)

  # Set the model dtype to bfloat16, and set the default device to the XLA device.
  # This will capture the model constructor into a graph so that we can add
  # sharding annotations to the weights later, and run the constructor on the XLA device.
  with set_default_dtype(model_dtype), torch_xla.device():
    model = initialize_model_class(config.model)

  n_params = sum([p.numel() for p in model.parameters()])
  logger.info(f"Training new model from scratch - Total size={n_params} params")

  # Downloading and loading a dataset from the hub.
  data = retry(
    lambda: make_huggingface_dataset(
      name=config.dataset_name,
      config_name=config.dataset_config_name,
      split="train",
      cache_dir=config.cache_dir,
      tokenizer=tokenizer,
      block_size=config.block_size,
    )
  )
  trainer = Trainer(
    model=model,
    config=config,
    train_dataset=data,
  )

  # TODO(https://github.com/pytorch/xla/issues/8954): Remove `jax_env_context`.
  with jax_env_context():
    trainer.train_loop(metrics_logger)


if __name__ == "__main__":
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
  )
  main()
