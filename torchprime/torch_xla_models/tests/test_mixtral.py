import copy
import unittest

import torch
import torch_xla
from omegaconf import OmegaConf
from transformers import AutoConfig
from transformers import MixtralForCausalLM as HfMixtralForCausalLM

from torchprime.torch_xla_models.mixtral import MixtralForCausalLM


class TestYourModule(unittest.TestCase):
  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    torch_xla.manual_seed(42)
    self.vocab_size = 128
    config = AutoConfig.from_pretrained(
      "mistralai/Mixtral-8x7B-v0.1",
      num_hidden_layers=1,
      num_attention_heads=8,
      hidden_size=8,
      intermediate_size=16,
      vocab_size=self.vocab_size,
    )
    config.attention_kernel = None
    torchprime_config = OmegaConf.create(
      {
        "vocab_size": 128,
        "hidden_size": 8,
        "intermediate_size": 16,
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
      self.hf_model = HfMixtralForCausalLM(config)
      self.model = MixtralForCausalLM(torchprime_config)
      self.model.load_state_dict(self.hf_model.state_dict())

  def test_forward_our_model_against_hf_model(self):
    device = torch_xla.device()
    model_xla = copy.deepcopy(self.model).to(device)
    hf_model_xla = copy.deepcopy(self.hf_model).to(device)
    torch_xla.sync()
    input_sizes = [8, 128, 256]
    for input_size in input_sizes:
      input = torch.randint(128, ((2, input_size // 2))).to(device)
      hf_output = hf_model_xla(
        input, labels=input, attention_mask=torch.ones_like(input)
      )
      mixtral_xla_logits, mixtral_xla_loss = model_xla(
        input, labels=input, attention_mask=torch.ones_like(input)
      )
      torch_xla.sync()
      self.assertTrue(
        torch.allclose(hf_output.logits, mixtral_xla_logits, atol=1e-6),
        "logits are not equal",
      )
      self.assertTrue(
        torch.allclose(hf_output.loss, mixtral_xla_loss, atol=1e-6),
        "loss is not equal",
      )

  def test_forward_torch_xla_against_native(self):
    input_size = 8
    device = torch.device("cpu")
    input = torch.randint(self.vocab_size, ((2, input_size // 2)))
    mixtral_native_logits, mixtral_native_loss = self.model(
      input, labels=input, attention_mask=torch.ones_like(input)
    )

    device = torch_xla.device()
    input = input.to(device)
    model_xla = copy.deepcopy(self.model).to(device)
    torch_xla.sync()

    mixtral_xla_logits, mixtral_xla_loss = model_xla(
      input, labels=input, attention_mask=torch.ones_like(input)
    )
    torch_xla.sync()
    self.assertTrue(
      torch.allclose(mixtral_native_logits, mixtral_xla_logits.to("cpu"), atol=1e-2),
      "CPU run and XLA run logits are not equal",
    )
    self.assertTrue(
      torch.allclose(mixtral_native_loss, mixtral_native_loss.to("cpu"), atol=1e-2),
      "CPU run and XLA run loss is not equal",
    )


if __name__ == "__main__":
  unittest.main()
