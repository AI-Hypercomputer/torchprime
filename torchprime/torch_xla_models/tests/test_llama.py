import copy
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch_xla
from omegaconf import OmegaConf
from transformers import AutoConfig
from transformers import LlamaForCausalLM as HfLlamaForCausalLM

from torchprime.torch_xla_models.model.llama import LlamaForCausalLM
from torchprime.torch_xla_models.tests.test_utils import (
  get_forward_and_backward_outputs,
)


@dataclass
class LlamaFixture:
  vocab_size: int
  hf_model: HfLlamaForCausalLM
  model: LlamaForCausalLM


def get_llama_3_8b() -> LlamaFixture:
  torch.manual_seed(42)
  torch_xla.manual_seed(42)
  vocab_size = 128
  config_path = Path(__file__).parent / "hf_model_config" / "meta-llama-3-8b"
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
      "vocab_size": 128,
      "hidden_size": 512,
      "intermediate_size": 64,
      "num_hidden_layers": 1,
      "num_attention_heads": 8,
      "num_key_value_heads": 8,
      "hidden_act": "silu",
      "max_position_embeddings": 8192,
      "initializer_range": 0.02,
      "rms_norm_eps": 1.0e-05,
      "attention_dropout": False,
      "attention_bias": False,
      "attention_kernel": None,
      "rope_theta": 500000.0,
    }
  )
  # Place model on CPU device first
  with torch.device("cpu"):
    hf_model = HfLlamaForCausalLM(config)
    model = LlamaForCausalLM(torchprime_config)
    model.load_state_dict(hf_model.state_dict())
  return LlamaFixture(vocab_size, hf_model, model)


def get_llama_3_1_405b() -> LlamaFixture:
  torch.manual_seed(42)
  torch_xla.manual_seed(42)
  vocab_size = 256
  config_path = Path(__file__).parent / "hf_model_config" / "meta-llama-3.1-405b"
  config = AutoConfig.from_pretrained(
    config_path,
    head_dim=64,
    num_hidden_layers=2,
    num_attention_heads=8,
    hidden_size=512,
    intermediate_size=64,
    vocab_size=vocab_size,
  )
  config.flash_attention = False
  torchprime_config = OmegaConf.create(
    {
      "vocab_size": 256,
      "hidden_size": 512,
      "intermediate_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 8,
      "num_key_value_heads": 8,
      "hidden_act": "silu",
      "max_position_embeddings": 131072,
      "initializer_range": 0.02,
      "rms_norm_eps": 1.0e-05,
      "attention_dropout": False,
      "attention_bias": False,
      "attention_kernel": None,
      "rope_theta": 500000.0,
      "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_len": 8192,
      },
    }
  )
  # Place model on CPU device first
  with torch.device("cpu"):
    hf_model = HfLlamaForCausalLM(config)
    model = LlamaForCausalLM(torchprime_config)
    # Assert that the `inv_freq` values are the same
    assert isinstance(model.model.layers[0].self_attn, nn.Module)
    assert isinstance(hf_model.model.layers[0].self_attn, nn.Module)
    assert isinstance(model.model.rotary_emb, nn.Module)
    assert isinstance(hf_model.model.rotary_emb, nn.Module)
    torch.testing.assert_close(
      model.model.rotary_emb.inv_freq,
      hf_model.model.rotary_emb.inv_freq,
    )
    # In this simplified model architecture, hidden_size 512 / num_attention_heads 8 = 64 head dim,
    # and the inv_freq size is half of the head dim.
    assert model.model.rotary_emb.inv_freq.shape == (32,)
    model.load_state_dict(hf_model.state_dict())
  return LlamaFixture(vocab_size, hf_model, model)


def noop(mod):
  return mod


def scan_decoders(mod):
  import torchprime.torch_xla_models.scan_layers

  return torchprime.torch_xla_models.scan_layers.compile(mod, "model.layers")


@pytest.mark.parametrize(
  "fixture",
  [get_llama_3_8b, get_llama_3_1_405b],
  ids=["Llama 3.0 8B", "Llama 3.1 405B"],
)
@pytest.mark.parametrize("transform", [noop, scan_decoders])
@pytest.mark.parametrize("input_size", [8, 128, 256])
def test_forward_and_backward_our_model_against_hf_model(
  fixture, transform, input_size
):
  """Compares the numerical consistency of our model and huggingface model.

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
  input_ids = torch.randint(fixture.vocab_size, ((2, input_size // 2))).to(device)
  attention_mask = torch.ones_like(input_ids)

  # Act
  (hf_logits, hf_loss), hf_params = get_forward_and_backward_outputs(
    hf_model_xla, input_ids=input_ids, labels=input_ids, attention_mask=attention_mask
  )
  (model_logits, model_loss), model_params = get_forward_and_backward_outputs(
    model_xla, input_ids=input_ids, labels=input_ids, attention_mask=attention_mask
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
    assert p_hf.grad is not None, f"Gradient for {name_hf} is None in hf_model"
    assert p_model.grad is not None, f"Gradient for {name_model} is None in model"
    torch.testing.assert_close(
      p_hf.grad,
      p_model.grad,
      atol=1e-6,
      rtol=1e-9,
      msg=f"Gradients for '{name_hf}' differ",
    )


@pytest.mark.parametrize(
  "fixture",
  [get_llama_3_8b, get_llama_3_1_405b],
  ids=["Llama 3.0 8B", "Llama 3.1 405B"],
)
@pytest.mark.parametrize("input_size", [8])
def test_forward_and_backward_torch_xla_against_native(fixture, input_size):
  """Compares the numerical consistency of our native and XLA models.

  Asserts that logits, loss, and gradients are nearly identical after a
  full forward and backward pass on both backends.
  """
  # Arrange
  fixture = fixture()
  cpu_device = torch.device("cpu")
  input_ids = torch.randint(
    fixture.vocab_size, ((2, input_size // 2)), device=cpu_device
  )
  attention_mask = torch.ones_like(input_ids)

  # Act
  # --- Native CPU pass ---
  model_native = fixture.model
  (logits_native, loss_native), params_native = get_forward_and_backward_outputs(
    model_native, input_ids=input_ids, labels=input_ids, attention_mask=attention_mask
  )

  # --- XLA pass ---
  xla_device = torch_xla.device()
  model_xla = copy.deepcopy(model_native).to(xla_device)
  input_ids_xla = input_ids.to(xla_device)
  attention_mask_xla = attention_mask.to(xla_device)

  (logits_xla, loss_xla), params_xla = get_forward_and_backward_outputs(
    model_xla,
    input_ids=input_ids_xla,
    labels=input_ids_xla,
    attention_mask=attention_mask_xla,
  )

  # Assert
  torch.testing.assert_close(
    logits_native,
    logits_xla.to("cpu"),
    atol=1e-2,
    rtol=1e-6,
    msg="CPU run and XLA run logits are not equal",
  )
  if loss_native is not None and loss_xla is not None:
    torch.testing.assert_close(
      loss_native,
      loss_xla.to("cpu"),
      atol=1e-2,
      rtol=1e-6,
      msg="Native run and XLA run loss is not equal",
    )

  for (name_native, p_native), (name_xla, p_xla) in zip(
    params_native, params_xla, strict=True
  ):
    assert name_native == name_xla
    assert p_native.grad is not None, (
      f"Gradient for {name_native} is None in native model"
    )
    assert p_xla.grad is not None, f"Gradient for {name_xla} is None in XLA model"
    torch.testing.assert_close(
      p_native.grad,
      p_xla.grad.cpu(),
      atol=1e-2,
      rtol=1e-6,
      msg=f"Gradients for '{name_native}' differ between Native and XLA",
    )
