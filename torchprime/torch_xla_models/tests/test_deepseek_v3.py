import copy
from dataclasses import dataclass

import pytest
import torch
import torch_xla
from omegaconf import OmegaConf
from transformers import AutoConfig
from transformers import DeepseekV3ForCausalLM as HFDeepseekV3ForCausalLM

from torchprime.torch_xla_models.model.deepseek_v3 import (
  DeepseekV3ForCausalLM,  # noqa: E402
)


@dataclass
class DeepseekFixture:
  vocab_size: int
  hf_model: HFDeepseekV3ForCausalLM
  model: DeepseekV3ForCausalLM


def get_deepseek_v3_dummy() -> DeepseekFixture:
  torch.manual_seed(42)
  torch_xla.manual_seed(42)
  vocab_size = 64
  config = AutoConfig.from_pretrained(
    "deepseek-ai/deepseek-v3",
  )
  config.vocab_size = vocab_size
  config.max_position_embeddings = vocab_size
  config.num_hidden_layers = 1

  scale_factor = 32
  config.attention_kernel="pytorch"

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

  tp_cfg = OmegaConf.create(config.to_dict())
  with torch.device("cpu"):
    hf_model = HFDeepseekV3ForCausalLM(config)
    hf_model.init_weights()
    model = DeepseekV3ForCausalLM(tp_cfg)
    model.load_state_dict(hf_model.state_dict())
  return DeepseekFixture(vocab_size, hf_model, model)


def noop(mod):
  return mod


def scan_decoders(mod):
  import torchprime.torch_xla_models.scan_layers

  return torchprime.torch_xla_models.scan_layers.compile(mod, "model.layers")


@pytest.mark.parametrize("transform", [noop, scan_decoders])
def test_forward_our_model_against_hf_model(transform):
  fixture = get_deepseek_v3_dummy()
  device = torch_xla.device()
  model_xla = copy.deepcopy(fixture.model).to(device)
  model_xla = transform(model_xla)
  hf_model_xla = copy.deepcopy(fixture.hf_model).to(device)
  torch_xla.sync()
  for input_size in [8, 16]:
    input_ids = torch.randint(fixture.vocab_size, (2, input_size // 2)).to(device)
    hf_output = hf_model_xla(
      input_ids, labels=input_ids, attention_mask=torch.ones_like(input_ids)
    )
    deepseek_xla_logits, deepseek_xla_loss = model_xla(
      input_ids, labels=input_ids, attention_mask=torch.ones_like(input_ids)
    )
    torch_xla.sync()
    torch.testing.assert_close(
      hf_output.logits,
      deepseek_xla_logits,
    atol=1e-2,
    rtol=1e-4,
      msg="logits are not equal",
    )
    torch.testing.assert_close(
      hf_output.loss,
      deepseek_xla_loss,
    atol=1e-2,
    rtol=1e-4,
      msg="loss is not equal",
    )


def test_forward_torch_xla_against_native():
  fixture = get_deepseek_v3_dummy()
  input_size = 8
  device = torch.device("cpu")
  input_ids = torch.randint(fixture.vocab_size, (2, input_size // 2))
  native_logits, native_loss = fixture.model(
    input_ids, labels=input_ids, attention_mask=torch.ones_like(input_ids)
  )

  device = torch_xla.device()
  input_ids = input_ids.to(device)
  model_xla = copy.deepcopy(fixture.model).to(device)
  torch_xla.sync()

  xla_logits, xla_loss = model_xla(
    input_ids, labels=input_ids, attention_mask=torch.ones_like(input_ids)
  )
  torch_xla.sync()
  torch.testing.assert_close(
    native_logits,
    xla_logits.to("cpu"),
    atol=1e-2,
    rtol=1e-4,
    msg="CPU run and XLA run logits are not equal",
  )
  torch.testing.assert_close(
    native_loss,
    xla_loss.to("cpu"),
    atol=1e-2,
    rtol=1e-4,
    msg="CPU run and XLA run loss is not equal",
  )
