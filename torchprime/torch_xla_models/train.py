# Standard library imports
import functools
import importlib
import logging
import math
import sys
from timeit import default_timer as timer

# Third-party library imports
import datasets
import hydra
import torch

# PyTorch XLA imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import transformers
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_xla.distributed.fsdp import checkpoint_module
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
  SpmdFullyShardedDataParallel as FSDPv2,
)

# Transformers imports
from transformers import (
  AutoTokenizer,
  default_data_collator,
  get_scheduler,
  set_seed,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.optimization import Adafactor
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.utils import check_min_version

from torchprime.metrics.step_duration import step_duration_from_latest_profile

check_min_version("4.39.3")
logger = logging.getLogger(__name__)

xr.use_spmd()
assert xr.is_spmd() is True


class Trainer:
  """The trainer."""

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
    num_devices = xr.global_runtime_device_count()
    assert num_devices == math.prod(
      [i for i in config.mesh.values()]
    ), "Mesh is not using all the available devices."
    dcn_mesh_shape = (config.mesh.dcn, 1, 1, 1)
    ici_mesh_shape = (1, config.mesh.fsdp, config.mesh.tensor, config.mesh.expert)
    mesh = xs.HybridMesh(
      ici_mesh_shape=ici_mesh_shape,
      dcn_mesh_shape=dcn_mesh_shape,
      axis_names=("dcn", "fsdp", "tensor", "expert"),
    )
    xs.set_global_mesh(mesh)
    logger.info(f"Logical mesh shape: {mesh.shape()}")
    # TODO (https://github.com/AI-Hypercomputer/torchprime/issues/66): Test this for multislice
    self.input_sharding_spec = xs.ShardingSpec(mesh, ("fsdp", None), minibatch=True)
    self.model = self._shard_model(model)

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
    sampler = torch.utils.data.DistributedSampler(
      self.train_dataset,
      num_replicas=num_replicas,
      rank=xr.process_index(),
    )
    assert self.global_batch_size is not None
    dataloader = DataLoader(
      self.train_dataset,
      # Data collator will default to DataCollatorWithPadding, so we change it.
      collate_fn=default_data_collator,
      # This is the host batch size.
      batch_size=self.global_batch_size // num_replicas,
      sampler=sampler,
      drop_last=True,
    )
    loader = pl.MpDeviceLoader(
      dataloader, self.device, input_sharding=self.input_sharding_spec
    )
    return loader

  def _shard_model(self, model):
    default_transformer_cls_names_to_wrap = []
    fsdp_transformer_layer_cls_to_wrap = self.config.model.fsdp.get(
      "transformer_layer_cls_to_wrap",
      default_transformer_cls_names_to_wrap,
    )

    transformer_cls_to_wrap = set()
    for layer_class in fsdp_transformer_layer_cls_to_wrap:
      transformer_cls = get_module_class_from_name(model, layer_class)
      if transformer_cls is None:
        raise Exception(
          "Could not find the transformer layer class to wrap in the model."
        )
      else:
        transformer_cls_to_wrap.add(transformer_cls)
    logger.info(f"Model classes to wrap: {transformer_cls_to_wrap}")
    auto_wrap_policy = functools.partial(
      transformer_auto_wrap_policy,
      # Transformer layer class to wrap
      transformer_layer_cls=transformer_cls_to_wrap,
    )

    if self.config.model.fsdp["xla_fsdp_grad_ckpt"]:
      # Apply gradient checkpointing to auto-wrapped sub-modules if specified
      logger.info("Enabling gradient checkpointing")

      def auto_wrapper_callable(m, *args, **kwargs):
        target_cls = FSDPv2
        return target_cls(checkpoint_module(m), *args, **kwargs)

    def shard_output(output, mesh):
      real_output = None
      if isinstance(output, torch.Tensor):
        real_output = output
      elif isinstance(output, tuple):
        real_output = output[0]
      elif isinstance(output, CausalLMOutputWithPast):
        real_output = output.logits
      if real_output is None:
        raise ValueError(
          "Something went wrong, the output of the model shouldn't be `None`"
        )
      # It is expected that the first dimension of the output is the batch size
      # which is usually sharded among all the devices except the tensor axis.
      xs.mark_sharding(real_output, mesh, (("dcn", "fsdp", "expert"), None, "tensor"))

    model = FSDPv2(
      model,
      shard_output=shard_output,
      auto_wrap_policy=auto_wrap_policy,
      auto_wrapper_callable=auto_wrapper_callable,
    )

    return model

  def train_loop(self):
    self.model.train()
    self.model.zero_grad()

    # For now we assume that we wil never train for mor than one epoch
    max_step = self.config.max_steps
    train_loader = self._get_train_dataloader()
    train_iterator = iter(train_loader)

    logger.info("Starting training")
    logger.info(f"    Max step: {max_step}")
    logger.info(f"    Global batch size: {self.global_batch_size}")

    for step in range(max_step):
      try:
        batch = next(train_iterator)
      except StopIteration:
        break

      trace_start_time = timer()
      loss = self.train_step(batch)
      trace_end_time = timer()

      if step % self.config.logging_steps == 0:

        def step_closure(step, loss, trace_start_time, trace_end_time):
          logger.info(
            f"Step: {step}, loss: {loss:0.4f}, "
            f"trace time: {(trace_end_time - trace_start_time) * 1000:0.2f} ms"
          )
          if math.isnan(loss):
            raise ValueError(f"Loss is NaN at step {step}")

        xm.add_step_closure(
          step_closure,
          args=(step, loss, trace_start_time, trace_end_time),
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

    # Analyze the step duration from the latest profile
    if self.config.profile_step >= 0:
      step_duration = step_duration_from_latest_profile(self.config.profile_dir)
      logger.info(f"Step duration: {step_duration:.3f} s")

  @torch_xla.compile(full_graph=True)
  def train_step(self, batch):
    _logits, loss = self.model(**batch)
    loss.backward()
    self.optimizer.step()
    self.lr_scheduler.step()
    self.model.zero_grad()
    return loss


def initialize_model_class(model_config):
  """Import and initalize model_class specified by the config."""
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


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: DictConfig):
  # Configure logging
  print(OmegaConf.to_yaml(config))  # Print the config for debugging
  log_level = logging.INFO
  logger.setLevel(log_level)
  logger.setLevel(log_level)
  datasets.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

  set_seed(config.seed)
  server = xp.start_server(9012)
  logger.info(f"Profiling server started: {str(server)}")

  # TODO: Add tokenizer models to torchprime
  tokenizer_name = config.model.tokenizer_name
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

  model = initialize_model_class(config.model)
  n_params = sum([p.numel() for p in model.parameters()])
  logger.info(f"Training new model from scratch - Total size={n_params} params")

  # Set the model dtype to bfloat16
  model = model.to(torch.bfloat16)

  # Downloading and loading a dataset from the hub.
  data = load_dataset(
    config.dataset_name,
    config.dataset_config_name,
    cache_dir=config.cache_dir,
  )["train"]
  column_names = list(data.features)
  data = data.map(
    lambda samples: tokenizer(samples["text"]),
    batched=True,
    remove_columns=column_names,
  )

  # Taken from run_clm.py. It's important to group texts evenly to avoid recompilations in TPU.
  block_size = config.block_size

  def group_texts(examples):
    from itertools import chain

    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
      k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
      for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

  data = data.map(group_texts, batched=True)

  trainer = Trainer(
    model=model,
    config=config,
    train_dataset=data,
  )

  trainer.train_loop()


if __name__ == "__main__":
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
  )
  main()
