"""Unit tests for model initialization using YAML configs.

This test scans all YAML files under the model config directory,
instantiates each model via the utility function, and checks:
- That the model loads successfully.
- That it is an instance of BaseCausalLM.
- That the number of trainable parameters exceeds a threshold.

This test can take a long time to run, depending on the number of models.
"""

import glob
import os

import pytest
import yaml
from omegaconf import OmegaConf

from torchprime.torch_xla_models.model.base_causal_lm import BaseCausalLM
from torchprime.torch_xla_models.model.model_utils import (
  extract_model_size_from_model_name,
  initialize_model_class,
)

MODEL_CONFIG_DIR = os.path.join("torchprime", "torch_xla_models", "configs", "model")


def get_model_config_files():
  models = glob.glob(os.path.join(MODEL_CONFIG_DIR, "*.yaml"))
  # only test models with size < 10B
  models = [m for m in models if 0 < extract_model_size_from_model_name(m) < 10]
  return models


@pytest.mark.parametrize("config_path", get_model_config_files())
def test_initialize_model_class_from_yaml(config_path):
  print(f"Testing model config: {config_path}")
  with open(config_path) as f:
    config_data = yaml.safe_load(f)

  config = OmegaConf.create(config_data)
  config.num_hidden_layers = 1  # Reduce for testing

  model = initialize_model_class(config)

  assert model is not None
  assert isinstance(model, BaseCausalLM)
  assert (
    sum(p.numel() for p in model.parameters()) > 1e6
  )  # 1e6 is some threshold for trainable params
