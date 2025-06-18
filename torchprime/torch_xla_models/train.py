import logging
import sys

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

from torchprime.data import DATASET_BUILDERS, make_train_dataset
from torchprime.metrics.metrics import MetricsLogger
from torchprime.torch_xla_models.model.model_utils import (
  initialize_model_class,
  log_parameter_breakdown,
  set_default_dtype,
)
from torchprime.torch_xla_models.trainer import TRAINERS, Trainer
from torchprime.utils.retry import retry

transformers.utils.check_min_version("4.39.3")
logger = logging.getLogger(__name__)

xr.use_spmd()
assert xr.is_spmd() is True


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: DictConfig):
  # Call metrics logger in the beginning to get the correct start time.
  metrics_logger = MetricsLogger()
  # Print the config for debugging
  print(OmegaConf.to_yaml(config))
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

  log_parameter_breakdown(model, logger)

  # Select dataset builder and trainer based on the task name.
  dataset_fn = DATASET_BUILDERS.get(config.task.name, make_train_dataset)
  trainer_cls = TRAINERS.get(config.task.name, Trainer)
  data = retry(lambda: dataset_fn(**config.dataset, tokenizer=tokenizer))

  dataset_name = getattr(config.dataset, "hf_dataset_name", None) or getattr(
    config.dataset, "file_dataset_path", "unknown"
  )
  logger.info("Loaded dataset `%s`, size=%d (packed) samples", dataset_name, len(data))

  trainer = trainer_cls(
    model=model,
    config=config,
    train_dataset=data,
  )

  # TODO(https://github.com/pytorch/xla/issues/8954): Remove `jax_env_context`.
  with jax_env_context():
    trainer.train_loop()
    trainer.finalize_training(metrics_logger)


if __name__ == "__main__":
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
  )
  main()
