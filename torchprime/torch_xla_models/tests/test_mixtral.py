import copy
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.test
import torch_xla
from omegaconf import OmegaConf
from transformers import AutoConfig
from transformers import MixtralForCausalLM as HfMixtralForCausalLM

from torchprime.torch_xla_models.model.mixtral import MixtralForCausalLM
from torchprime.torch_xla_models.tests.test_utils import (
  get_forward_and_backward_outputs,
)


@dataclass
class MixtralFixture:
  vocab_size: int
  hf_model: HfMixtralForCausalLM
  model: MixtralForCausalLM


def get_mixtral_8x7b() -> MixtralFixture:
  torch.manual_seed(42)
  torch_xla.manual_seed(42)
  vocab_size = 128
  config_path = Path(__file__).parent / "hf_model_config" / "mixtral-8x7b-v0.1"
  config = AutoConfig.from_pretrained(
    config_path,
    head_dim=64,
    num_hidden_layers=1,
    num_attention_heads=8,
    hidden_size=512,
    intermediate_size=64,
    vocab_size=vocab_size,
  )
  config.flash_attention = False
  torchprime_config = OmegaConf.create(
    {
      "vocab_size": vocab_size,
      "hidden_size": 512,
      "intermediate_size": 64,
      "num_hidden_layers": 1,
      "num_attention_heads": 8,
      "num_key_value_heads": 8,
      "max_position_embeddings": 32768,
      "initializer_range": 0.02,
      "rms_norm_eps": 1.0e-05,
      "num_experts_per_tok": 2,
      "num_local_experts": 8,
      "rope_theta": 1000000.0,
      "router_aux_loss_coef": 0.02,
      "attention_dropout": 0.0,
      "attention_bias": False,
      "attention_kernel": None,
      "moe_implementation": "static",
    }
  )
  # place model on CPU device first
  with torch.device("cpu"):
    hf_model = HfMixtralForCausalLM(config)
    model = MixtralForCausalLM(torchprime_config)
    model.load_state_dict(hf_model.state_dict())

  return MixtralFixture(vocab_size, hf_model, model)


def noop(mod):
  return mod


def scan_decoders(mod):
  import torchprime.torch_xla_models.scan_layers

  return torchprime.torch_xla_models.scan_layers.compile(mod, "model.layers")


@pytest.mark.parametrize(
  "fixture",
  [get_mixtral_8x7b],
  ids=["Mixtral 8x7B"],
)
@pytest.mark.parametrize("transform", [noop, scan_decoders])
@pytest.mark.parametrize("input_size", [8, 128, 256])
def test_forward_and_backward_our_model_against_hf_model(
  fixture, transform, input_size
):
  """Compares the numerical consistency of our Mixtral model against the
  Hugging Face reference on an XLA device.

  Asserts that logits, loss, and gradients are nearly identical after a
  full forward and backward pass.
  """
  # Arrange
  fixture = fixture()
  device = torch_xla.device()
  model_xla = copy.deepcopy(fixture.model).to(device)
  model_xla = transform(model_xla)
  hf_model_xla = copy.deepcopy(fixture.hf_model).to(device)
  torch_xla.sync()
  input_ids = torch.randint(128, ((2, input_size // 2))).to(device)
  attention_mask = torch.ones_like(input_ids)

  # Act
  (hf_logits, hf_loss), hf_params = get_forward_and_backward_outputs(
    hf_model_xla,
    input_ids=input_ids,
    labels=input_ids,
    attention_mask=attention_mask,
  )
  (model_logits, model_loss), model_params = get_forward_and_backward_outputs(
    model_xla,
    input_ids=input_ids,
    labels=input_ids,
    attention_mask=attention_mask,
  )

  # Assert
  torch.testing.assert_close(
    hf_logits, model_logits, atol=1e-6, rtol=1e-9, msg="Logits are not equal"
  )
  torch.testing.assert_close(
    hf_loss, model_loss, atol=1e-6, rtol=1e-9, msg="Losses are not equal"
  )
  for (name_hf, p_hf), (name_model, p_model) in zip(
    hf_params, model_params, strict=True
  ):
    assert name_hf == name_model, f"Parameter name mismatch: {name_hf} vs {name_model}"
    assert p_model.grad is not None, (
      f"Model grad is None for {name_model} when HF grad is not None"
    )
    if p_hf.grad is None:
      # Gradient value for unused parameter is None in hf model while our model have 0
      assert torch.all(p_model.grad == 0), (
        f"Model grad for {name_model} is not zero when HF grad is None"
      )
    else:
      torch.testing.assert_close(
        p_hf.grad,
        p_model.grad,
        atol=1e-6,
        rtol=1e-9,
        msg=f"Gradients for '{name_hf}' differ",
      )


@pytest.mark.parametrize(
  "fixture",
  [get_mixtral_8x7b],
  ids=["Mixtral 8x7B"],
)
def test_forward_and_backward_torch_xla_against_native(fixture):
  """Compares the numerical consistency of our Mixtral model on native CPU
  vs. an XLA device.

  Asserts that logits, loss, and gradients are nearly identical after a
  full forward and backward pass on both backends.
  """
  # Arrange
  fixture = fixture()
  input_size = 8
  device = torch.device("cpu")
  input_ids = torch.randint(fixture.vocab_size, ((2, input_size // 2)), device=device)
  attention_mask = torch.ones_like(input_ids)

  # Act
  (logits_native, loss_native), params_native = get_forward_and_backward_outputs(
    fixture.model,
    input_ids=input_ids,
    labels=input_ids,
    attention_mask=attention_mask,
  )
  (logits_xla, loss_xla), params_xla = get_forward_and_backward_outputs(
    copy.deepcopy(fixture.model).to(torch_xla.device()),
    input_ids=input_ids.to(torch_xla.device()),
    labels=input_ids.to(torch_xla.device()),
    attention_mask=attention_mask.to(torch_xla.device()),
  )

  # Assert
  torch.testing.assert_close(
    logits_native,
    logits_xla.to("cpu"),
    atol=1e-2,
    rtol=1e-4,
    msg="CPU run and XLA run logits are not equal",
  )
  torch.testing.assert_close(
    loss_native,
    loss_xla.to("cpu"),
    atol=1e-2,
    rtol=1e-4,
    msg="CPU run and XLA run loss is not equal",
  )
  for (name_native, p_native), (name_xla, p_xla) in zip(
    params_native, params_xla, strict=True
  ):
    assert name_native == name_xla
    assert p_native.grad is not None
    assert p_xla.grad is not None
    torch.testing.assert_close(
      p_native.grad,
      p_xla.grad.cpu(),
      atol=1e-2,
      rtol=1e-4,
      msg=f"Gradients for '{name_native}' differ between Native and XLA",
    )
