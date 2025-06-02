import importlib
import logging
import sys
from contextlib import contextmanager

import datasets
import hydra
import torch
import torch_xla
import torch_xla.debug.profiler as xp
import torch_xla.runtime as xr
import transformers
from omegaconf import DictConfig, OmegaConf
from torch_xla._internal.jax_workarounds import jax_env_context
from transformers import (
  AutoTokenizer,
  set_seed,
)

from torchprime.data.dataset import make_huggingface_dataset
from torchprime.metrics.metrics import MetricsLogger
from torchprime.torch_xla_models.trainer.base_trainer import Trainer
from torchprime.utils.retry import retry

transformers.utils.check_min_version("4.39.3")
logger = logging.getLogger(__name__)

xr.use_spmd()
assert xr.is_spmd() is True


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
