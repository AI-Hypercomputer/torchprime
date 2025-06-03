"""Test for loading Meta-LLaMA-3-8B from HuggingFace.

This test verifies that a model initialized from a local YAML config can
successfully load pretrained weights from Hugging Face and that the number of
parameters matches expectations.

It also ensures that if the config is mutated to be incompatible with the
checkpoint (e.g., changing the number of layers), the model will raise an error
during weight loading.
"""

import os
import sys

import pytest
import yaml
from omegaconf import OmegaConf

from torchprime.torch_xla_models.model.base_causal_lm import BaseCausalLM
from torchprime.torch_xla_models.model.model_utils import initialize_model_class

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../model")))


@pytest.mark.integration
def test_llama3_8b_from_pretrained_param_count():
  with open(
    os.path.join(
      "torchprime", "torch_xla_models", "configs", "model", "llama-3-8b.yaml"
    )
  ) as f:
    config_data = yaml.safe_load(f)

  config = OmegaConf.create(config_data)
  model = initialize_model_class(config)
  random_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  assert isinstance(model, BaseCausalLM)

  try:
    model.from_pretrained("meta-llama/Meta-Llama-3-8B")
  except Exception as e:
    pytest.fail(f"Failed to load Meta-Llama-3-8B: {e}")

  model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

  assert random_model_params == model_params, "Unmatched number of parameters"

  # Modify config to break the architecture
  config.num_hidden_layers = config.num_hidden_layers - 1
  mismatched_model = initialize_model_class(config)

  # Expect state_dict loading to fail due to size/shape mismatch
  with pytest.raises(RuntimeError, match="size mismatch|missing|unexpected"):
    mismatched_model.from_pretrained("meta-llama/Meta-Llama-3-8B")
