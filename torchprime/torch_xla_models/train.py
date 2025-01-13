# Standard library imports
import functools
import logging
import os
import sys
from dataclasses import dataclass, field
from timeit import default_timer as timer

# Third-party library imports
import datasets
import numpy as np
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
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_xla.distributed.fsdp import checkpoint_module
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
    SpmdFullyShardedDataParallel as FSDPv2,
)  # yapf: disable

# Transformers imports
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.utils import check_min_version

# TorchPrime imports
from torchprime.torch_xla_models.llama import LlamaForCausalLM

check_min_version("4.39.3")
logger = logging.getLogger(__name__)

xr.use_spmd()
assert xr.is_spmd() is True


@dataclass
class ModelArguments:
  """
  Arguments pertaining to which model/config/tokenizer we are going to fine-tune,
  or train from scratch.
  """

  model_id: str | None = field(
      default="meta-llama/Llama-2-7b-hf",
      metadata={
          "help": "Pretrained config name or path if not the same as model_name"
      },
  )
  tokenizer_name: str | None = field(
      default=None,
      metadata={"help": "Name of the tokenizer if different from model_id"},
  )
  cache_dir: str | None = field(
      default=None,
      metadata={
          "help":
              "Where do you want to store the pretrained models downloaded from huggingface.co"
      },
  )


@dataclass
class MoreTrainingArguments(TrainingArguments):
  profile_step: int | None = field(
      default=-1, metadata={"help": "Step to profile"})
  profile_dir: str | None = field(
      default="profile/", metadata={"help": "Directory to store the profile"})
  profile_duration: int | None = field(
      default=20000, metadata={"help": "Duration (ms) to capture profile"})
  global_batch_size: int | None = field(
      default=None, metadata={"help": "Global batch size"})


@dataclass
class DataTrainingArguments:
  """
  Arguments pertaining to what data we are going to input our model for training.
  """

  dataset_name: str | None = field(
      default=None,
      metadata={
          "help": "The name of the dataset to use (via the datasets library)."
      },
  )
  dataset_config_name: str | None = field(
      default=None,
      metadata={
          "help":
              "The configuration name of the dataset to use (via the datasets library)."
      },
  )
  block_size: int | None = field(
      default=None,
      metadata={
          "help": (
              "Optional input sequence length after tokenization. "
              "The training dataset will be truncated in block of this size for training. "
              "Default to the model max input length for single sentence inputs (take into account special tokens)."
          )
      },
  )

  def __post_init__(self):
    if (self.dataset_name is None and self.train_file is None and
        self.validation_file is None):
      raise ValueError(
          "Need either a dataset name or a training/validation file.")


class Trainer:
  """The trainer."""

  def __init__(
      self,
      model: nn.Module,
      args: MoreTrainingArguments,
      train_dataset: Dataset | IterableDataset | None,
  ):
    self.args = args
    self.device = xm.xla_device()
    self.global_batch_size = args.global_batch_size
    self.train_dataset = train_dataset

    # Set up SPMD mesh and shard the model
    num_devices = xr.global_runtime_device_count()
    xs.set_global_mesh(
        xs.Mesh(
            np.array(range(num_devices)),
            (num_devices, 1),
            axis_names=("fsdp", "tensor"),
        ))
    logger.info(f"Logical mesh shape: {xs.get_global_mesh().shape()}")
    self.input_sharding_spec = xs.ShardingSpec(
        xs.get_global_mesh(), ("fsdp", None), minibatch=True)
    self.model = self._shard_model(model)

    # Set up optimizers
    self.optimizer = AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )

    # TODO: this OOMs the TPU.
    # self._prime_optimizer()

    self.lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=self.optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
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
        dataloader, self.device, input_sharding=self.input_sharding_spec)
    return loader

  def _shard_model(self, model):
    default_transformer_cls_names_to_wrap = []
    fsdp_transformer_layer_cls_to_wrap = self.args.fsdp_config.get(
        "transformer_layer_cls_to_wrap",
        default_transformer_cls_names_to_wrap,
    )

    transformer_cls_to_wrap = set()
    for layer_class in fsdp_transformer_layer_cls_to_wrap:
      transformer_cls = get_module_class_from_name(model, layer_class)
      if transformer_cls is None:
        raise Exception(
            "Could not find the transformer layer class to wrap in the model.")
      else:
        transformer_cls_to_wrap.add(transformer_cls)
    logger.info(f"Llama classes to wrap: {transformer_cls_to_wrap}")
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        # Transformer layer class to wrap
        transformer_layer_cls=transformer_cls_to_wrap,
    )

    if self.args.fsdp_config["xla_fsdp_grad_ckpt"]:
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
            "Something went wrong, the output of the model shouldn't be `None`")
      xs.mark_sharding(real_output, mesh, ("fsdp", None, None))

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
    max_step = self.args.max_steps
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

      # For logging step, we explicitly isolate this step from tracing and execution overlapping.
      if step % self.args.logging_steps == 0:
        xm.wait_device_ops()
      trace_start_time = timer()

      loss = self.train_step(batch)

      trace_end_time = timer()
      if step % self.args.logging_steps == 0:
        xm.wait_device_ops()
        execute_end_time = timer()
        logger.info(
            f"Step: {step}, loss: {loss:0.4f}, trace time: {(trace_end_time - trace_start_time) * 1000:0.2f} ms, step time: {(execute_end_time - trace_end_time) * 1000:0.2f} ms"
        )

      # Capture profile at the prefer step
      if step == self.args.profile_step:
        # Wait until device execution catches up to tracing before triggering the profile. This will
        # interrupt training slightly on the hosts which are capturing, but by waiting after tracing
        # for the step, the interruption will be minimal.
        xm.wait_device_ops()
        xp.trace_detached(
            "127.0.0.1:9012",
            self.args.profile_dir,
            self.args.profile_duration,
        )

    logger.info("Finished training run")

  @torch_xla.compile(full_graph=True)
  def train_step(self, batch):
    outputs = self.model(**batch)
    loss = outputs.loss
    loss.backward()
    self.optimizer.step()
    self.lr_scheduler.step()
    self.model.zero_grad()
    return loss


def main():
  # Parse CLI arguments
  parser = HfArgumentParser(
      (ModelArguments, DataTrainingArguments, MoreTrainingArguments))

  if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(
        json_file=os.path.abspath(sys.argv[1]))
  else:
    model_args, data_args, training_args = (
        parser.parse_args_into_dataclasses())

  # Configure logging
  if training_args.should_log:
    # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    transformers.utils.logging.set_verbosity_info()

  log_level = training_args.get_process_log_level()
  logger.setLevel(log_level)
  datasets.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

  set_seed(training_args.seed)
  server = xp.start_server(9012)
  logger.info(f"Profiling server started: {str(server)}")

  tokenizer_name = (
      model_args.tokenizer_name
      if model_args.tokenizer_name else model_args.model_id)
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  config = AutoConfig.from_pretrained(model_args.model_id)
  config.flash_attention = True
  model = LlamaForCausalLM(config)
  n_params = sum([p.numel() for p in model.parameters()])
  logger.info(f"Training new model from scratch - Total size={n_params} params")

  # Set the model dtype to bfloat16
  model = model.to(torch.bfloat16)

  # Downloading and loading a dataset from the hub.
  data = load_dataset(
      data_args.dataset_name,
      data_args.dataset_config_name,
      cache_dir=model_args.cache_dir,
  )["train"]
  column_names = list(data.features)
  data = data.map(
      lambda samples: tokenizer(samples["text"]),
      batched=True,
      remove_columns=column_names,
  )

  # Taken from run_clm.py. It's important to group texts evenly to avoid recompilations in TPU.
  block_size = data_args.block_size

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
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

  data = data.map(group_texts, batched=True)

  trainer = Trainer(
      model=model,
      args=training_args,
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
