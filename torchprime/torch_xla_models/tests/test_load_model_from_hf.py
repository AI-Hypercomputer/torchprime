"""Test for loading Meta-LLaMA-3-8B from HuggingFace.

This test verifies that a model initialized from a local YAML config can
successfully load pretrained weights from Hugging Face and that the number of
parameters matches expectations.

It also ensures that if the config is mutated to be incompatible with the
checkpoint (e.g., changing the number of layers), the model will raise an error
during weight loading.
"""

import os

import pytest
import yaml
from omegaconf import OmegaConf

from torchprime.torch_xla_models.model.base_causal_lm import BaseCausalLM
from torchprime.torch_xla_models.model.model_utils import initialize_model_class


@pytest.mark.integration
@pytest.mark.parametrize(
  "config_file, hf_model, skip_on_ci",
  [
    (
      "llama-1b-random-for-test.yaml",
      "hf-internal-testing/tiny-random-LlamaForCausalLM",
      False,
    ),
    (
      "llama-3-8b.yaml",
      "meta-llama/Meta-Llama-3-8B",
      True,
    ),
  ],
)
def test_llama3_8b_from_pretrained_param_count(config_file, hf_model, skip_on_ci):
  if skip_on_ci and os.environ.get("CI"):  # set export CI=true in GitHub Actions
    pytest.skip(f"Skipping {hf_model} test in CI due to resource limits.")

  config_path = os.path.join(
    "torchprime", "torch_xla_models", "configs", "model", config_file
  )
  with open(config_path) as f:
    config_data = yaml.safe_load(f)

  config = OmegaConf.create(config_data)
  model = initialize_model_class(config)
  random_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  assert isinstance(model, BaseCausalLM)

  try:
    model.from_pretrained(hf_model)
  except Exception as e:
    pytest.fail(f"Failed to load weights for {config_file}: {e}")

  model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

  assert random_model_params == model_params, f"Param count mismatch in {config_file}"

  # Modify config to break the architecture
  config.num_hidden_layers -= 1
  mismatched_model = initialize_model_class(config)

  # Expect state_dict loading to fail due to size/shape mismatch
  with pytest.raises(RuntimeError, match=r"(Unexpected|Missing) key|size mismatch"):
    mismatched_model.from_pretrained(hf_model)
