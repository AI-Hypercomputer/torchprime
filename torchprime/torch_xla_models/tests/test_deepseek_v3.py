import copy
from dataclasses import dataclass

import pytest
import torch
import torch_xla
from omegaconf import OmegaConf
from transformers import DeepseekV3Config
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
  config = DeepseekV3Config(
    vocab_size=vocab_size,
    hidden_size=128,
    intermediate_size=256,
    moe_intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=64,
    use_cache=False,
  )
  tp_cfg = OmegaConf.create(config.to_dict())
  with torch.device("cpu"):
    hf_model = HFDeepseekV3ForCausalLM(config)
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
      atol=1e-6,
      rtol=1e-9,
      msg="logits are not equal",
    )
    torch.testing.assert_close(
      hf_output.loss,
      deepseek_xla_loss,
      atol=1e-6,
      rtol=1e-9,
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
    rtol=1e-6,
    msg="CPU run and XLA run logits are not equal",
  )
  torch.testing.assert_close(
    native_loss,
    xla_loss.to("cpu"),
    atol=1e-2,
    rtol=1e-6,
    msg="CPU run and XLA run loss is not equal",
  )
