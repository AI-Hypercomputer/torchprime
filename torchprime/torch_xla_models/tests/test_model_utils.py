"""Unit tests for model initialization using YAML configs."""

import glob
import os

import pytest
import yaml
from omegaconf import OmegaConf

from torchprime.torch_xla_models.model import model_utils
from torchprime.torch_xla_models.model.base_causal_lm import BaseCausalLM

MODEL_CONFIG_DIR = os.path.join("torchprime", "torch_xla_models", "configs", "model")
MAX_MODEL_SIZE = 10  # Only test models with size < 10B


def get_model_config_files():
  models = glob.glob(os.path.join(MODEL_CONFIG_DIR, "*.yaml"))
  models = [
    m
    for m in models
    if 0 < model_utils.extract_model_size_from_model_name(m) < MAX_MODEL_SIZE
  ]
  return models


@pytest.mark.parametrize("config_path", get_model_config_files())
def test_initialize_model_class_from_yaml(config_path):
  """This test scans all YAML files under the model config directory,
  instantiates each model via the utility function, and checks:
  - That the model loads successfully.
  - That it is an instance of BaseCausalLM.
  - That the number of trainable parameters exceeds a threshold.

  This test can take a long time to run, depending on the number of models.
  """
  print(f"Testing model config: {config_path}")
  with open(config_path) as f:
    config_data = yaml.safe_load(f)

  config = OmegaConf.create(config_data)
  config.num_hidden_layers = 1  # Reduce for testing

  model = model_utils.initialize_model_class(config)

  assert model is not None
  assert isinstance(model, BaseCausalLM)
  assert (
    sum(p.numel() for p in model.parameters()) > 1e6
  )  # 1e6 is some threshold for trainable params


@pytest.mark.parametrize(
  ("name", "expected"),
  [
    # Transformer blocks
    ("model.layers.0.self_attn.q_proj.weight", "layers_0"),
    ("encoder.layers.12.mlp.fc1.weight", "layers_12"),
    # CNN / ResNet
    ("layer1.0.conv1.weight", "layer1_0"),
    ("layer4.2.bn3.weight", "layer4_2"),
    # Diffusion UNet
    ("down_blocks.1.resnets.0.conv1.weight", "down_blocks_1"),
    # DataParallel prefix
    ("module.layer2.1.bn3.weight", "layer2_1"),
    # Misc single-scope params
    ("lm_head.weight", "lm_head"),
    ("bias", "bias"),
  ],
)
def test_get_param_group_key(name, expected):
  assert model_utils.get_param_group_key(name) == expected, f"{name} ➜ {expected}"
