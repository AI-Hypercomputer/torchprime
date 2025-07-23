"""Train script for LLMs using PyTorch/XLA with some torchax for lowering."""

import logging
import sys
from timeit import default_timer as timer

import datasets
import hydra
import omegaconf
import torch
import torch_xla
import torch_xla.debug.profiler as xp
import torch_xla.runtime as xr
import transformers

import torchprime.data
import torchprime.torch_xla_models.trainer
from torchprime.metrics import metrics
from torchprime.torch_xla_models.model import model_utils
from torchprime.utils import retry

transformers.utils.check_min_version("4.39.3")
logger = logging.getLogger(__name__)

xr.use_spmd()
assert xr.is_spmd() is True


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):
  # Call metrics logger in the beginning to get the correct start time.
  metrics_logger = metrics.MetricsLogger()

  # Print the config for debugging
  print(omegaconf.OmegaConf.to_yaml(config))
  log_level = logging.INFO
  logger.setLevel(log_level)
  datasets.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

  transformers.set_seed(config.seed)
  torch_xla.manual_seed(config.seed)
  server = xp.start_server(9012)
  logger.info(f"Profiling server started: {str(server)}")

  # TODO(https://github.com/AI-Hypercomputer/torchprime/issues/14): Add tokenizers to torchprime.
  tokenizer_name = config.model.tokenizer_name
  tokenizer = retry.retry(
    lambda: transformers.AutoTokenizer.from_pretrained(tokenizer_name)
  )

  assert config.torch_dtype == "bfloat16", "Currently only bfloat16 is supported"
  model_dtype = getattr(torch, config.torch_dtype)

  # Set the model dtype to bfloat16, and set the default device to the XLA device.
  # This will capture the model constructor into a graph so that we can add
  # sharding annotations to the weights later, and run the constructor on the XLA device.
  with model_utils.set_default_dtype(model_dtype), torch_xla.device():
    model = model_utils.initialize_model_class(config.model)

  model_utils.log_parameter_breakdown(model, logger)

  # Select dataset builder and trainer based on the task name.
  dataset_fn = torchprime.data.DATASET_BUILDERS.get(
    config.task.name, torchprime.data.make_train_dataset
  )

  trainer_cls = torchprime.torch_xla_models.trainer.TRAINERS.get(
    config.task.name, torchprime.torch_xla_models.trainer.Trainer
  )

  load_time_start = timer()
  data = retry.retry(lambda: dataset_fn(**config.dataset, tokenizer=tokenizer))
  load_time_end = timer()
  load_time_seconds = load_time_end - load_time_start

  dataset_name = getattr(config.dataset, "hf_dataset_name", None) or getattr(
    config.dataset, "file_dataset_path", "unknown"
  )
  num_tokens = len(data) * config.dataset.block_size
  tokens_per_second = num_tokens / load_time_seconds
  logger.info("--- Dataset Loading Benchmark ---")
  logger.info("  Dataset: %s", dataset_name)
  logger.info("  Num samples: %d", len(data))
  logger.info("  Total tokens: %d", num_tokens)
  logger.info(f"  Load time: {load_time_seconds:.2f} seconds")
  logger.info(f"  Tokens/sec: {tokens_per_second:,.2f}")
  logger.info("---------------------------------")
  metrics_logger.log_dataset_load_time(load_time_seconds)

  trainer = trainer_cls(
    model=model,
    config=config,
    train_dataset=data,
  )

  # TODO(https://github.com/pytorch/xla/issues/8954): Remove `jax_env_context`.
  with torch_xla._internal.jax_workarounds.jax_env_context():
    trainer.train_loop()
    trainer.finalize_training(metrics_logger)

  return 0


if __name__ == "__main__":
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
  )
  sys.exit(main())
