import copy
from dataclasses import dataclass

import pytest
import torch
import torch_xla
from omegaconf import OmegaConf
from transformers import AutoConfig
from transformers import DeepseekV3ForCausalLM as HFDeepseekV3ForCausalLM

from torchprime.torch_xla_models.model.deepseek_v3 import (
  DeepseekV3ForCausalLM,
  convert_hf_state_dict_for_grouped_moe,
)
from torchprime.torch_xla_models.tests.test_utils import (
  get_forward_and_backward_outputs,
)

MOE_START_FROM_LAYER = 2  # layer 0,1 dense layers and layer 2+ moe layers


@dataclass
class DeepseekFixture:
  vocab_size: int
  hf_model: HFDeepseekV3ForCausalLM
  model: DeepseekV3ForCausalLM


def get_deepseek_v3_dummy() -> DeepseekFixture:
  seed = 123
  torch.manual_seed(seed)
  torch_xla.manual_seed(seed)
  vocab_size = 64
  config = AutoConfig.from_pretrained(
    "deepseek-ai/deepseek-v3",
  )
  config.vocab_size = vocab_size
  config.max_position_embeddings = vocab_size
  config.first_k_dense_replace = MOE_START_FROM_LAYER
  config.num_hidden_layers = 5  # from 61
  config.n_group = 4  # from 8

  scale_factor = 32
  config.attention_kernel = "pytorch"

  config.hidden_size //= scale_factor
  config.intermediate_size //= scale_factor
  config.moe_intermediate_size //= scale_factor
  config.num_attention_heads //= scale_factor
  config.n_routed_experts //= scale_factor
  config.kv_lora_rank //= scale_factor
  config.q_lora_rank //= scale_factor
  config.qk_rope_head_dim //= scale_factor
  config.v_head_dim //= scale_factor
  config.qk_nope_head_dim //= scale_factor
  config.qk_head_dim //= scale_factor
  config.head_dim //= scale_factor
  config.num_key_value_heads //= scale_factor
  config.capacity_factor = 10.0

  tp_cfg = OmegaConf.create(config.to_dict())
  with torch.device("cpu"):
    hf_model = HFDeepseekV3ForCausalLM(config)
    hf_model.init_weights()
    hf_dict = hf_model.state_dict()

    model = DeepseekV3ForCausalLM(tp_cfg)
    converted_dict = convert_hf_state_dict_for_grouped_moe(hf_dict, model.config)
    model.load_state_dict(converted_dict, strict=True)

  return DeepseekFixture(vocab_size, hf_model, model)


def noop(mod):
  return mod


def scan_decoders(mod):
  import torchprime.torch_xla_models.scan_layers

  return torchprime.torch_xla_models.scan_layers.compile(
    mod, "model.layers", MOE_START_FROM_LAYER
  )


@pytest.mark.parametrize("transform", [noop, scan_decoders])
@pytest.mark.parametrize("input_size", [8, 16])
def test_forward_and_backward_our_model_against_hf_model(transform, input_size):
  """Compares the numerical consistency of our DeepseekV3 model against the
  Hugging Face reference on an XLA device.

  Asserts that logits, loss, and gradients are nearly identical after a
  full forward and backward pass.
  """
  # Arrange
  fixture = get_deepseek_v3_dummy()
  device = torch_xla.device()
  model_xla = copy.deepcopy(fixture.model).to(device)
  model_xla = transform(model_xla)
  hf_model_xla = copy.deepcopy(fixture.hf_model).to(device)
  torch_xla.sync()
  input_ids = torch.randint(fixture.vocab_size, (2, input_size // 2)).to(device)
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
    hf_logits, model_logits, atol=1e-2, rtol=1e-6, msg="Logits are not equal"
  )
  torch.testing.assert_close(
    hf_loss, model_loss, atol=1e-2, rtol=1e-6, msg="Losses are not equal"
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
      atol=1e-2,
      rtol=1e-4,
      msg=f"Gradients for '{name_hf}' differ. hf_grad: {p_hf.grad}, model_grad: {p_model.grad}",
    )

@pytest.mark.parametrize("transform", [noop, scan_decoders])
def test_layers_by_layer_against_hf_model(transform):
  fixture = get_deepseek_v3_dummy()
  device = torch_xla.device()
  model_xla = copy.deepcopy(fixture.model).to(device)
  model_xla = transform(model_xla)
  hf_model_xla = copy.deepcopy(fixture.hf_model).to(device)

  seq_len = 4
  input_ids = torch.randint(fixture.vocab_size, (2, seq_len)).to(device)
  attention_mask = torch.ones_like(input_ids)

  inputs_embeds_xla = model_xla.model.embed_tokens(input_ids)
  inputs_embeds_hf = hf_model_xla.model.embed_tokens(input_ids)
  torch.testing.assert_close(
    inputs_embeds_xla,
    inputs_embeds_hf,
    atol=1e-2,
    rtol=1e-6,
    msg="emb layer outputs not equal",
  )

  position_ids = torch.arange(seq_len, device=device).unsqueeze(0).float()
  causal_mask = (
    torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    .unsqueeze(0)
    .unsqueeze(0)
  )
  causal_mask = causal_mask * attention_mask[:, None, None, :]

  pos_embeds_xla = model_xla.model.rotary_emb(inputs_embeds_xla, position_ids)
  pos_embeds_hf = hf_model_xla.model.rotary_emb(inputs_embeds_hf, position_ids)
  torch.testing.assert_close(
    pos_embeds_xla[0],
    pos_embeds_hf[0],
    atol=1e-2,
    rtol=1e-6,
    msg="rotary_emb layer outputs not equal",
  )
  torch.testing.assert_close(
    pos_embeds_xla[1],
    pos_embeds_hf[1],
    atol=1e-2,
    rtol=1e-6,
    msg="rotary_emb layer outputs not equal",
  )

  hidden_xla = inputs_embeds_xla
  hidden_hf = inputs_embeds_hf
  for idx, (layer_xla, layer_hf) in enumerate(
    zip(model_xla.model.layers, hf_model_xla.model.layers, strict=True)
  ):
    hidden_xla = layer_xla(
      hidden_xla,
      attention_mask=causal_mask,
      position_ids=position_ids,
      position_embeddings=pos_embeds_xla,
    )
    hidden_hf = layer_hf(
      hidden_hf,
      attention_mask=causal_mask,
      position_ids=position_ids,
      position_embeddings=pos_embeds_hf,
    )[0]
    torch_xla.sync()
    torch.testing.assert_close(
      hidden_xla,
      hidden_hf,
      atol=1e-2,
      rtol=1e-6,
      msg=f"decoder layer {idx} outputs not equal",
    )


def test_forward_and_backward_torch_xla_against_native_cpu():
  """Compares the numerical consistency of our DeepseekV3 model on native CPU
  vs. an XLA device.

  Asserts that logits, loss, and gradients are nearly identical after a
  full forward and backward pass on both backends.
  """
  # Arrange
  fixture = get_deepseek_v3_dummy()
  input_size = 8
  cpu_device = torch.device("cpu")
  input_ids = torch.randint(fixture.vocab_size, (2, input_size // 2), device=cpu_device)
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
    rtol=1e-6,
    msg="CPU run and XLA run logits are not equal",
  )
  torch.testing.assert_close(
    loss_native,
    loss_xla.to("cpu"),
    atol=1e-2,
    rtol=1e-6,
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
