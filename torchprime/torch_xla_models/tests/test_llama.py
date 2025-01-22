import copy
import unittest

import torch
import torch_xla
from transformers import AutoConfig
from transformers import LlamaForCausalLM as HfLlamaForCausalLM

from torchprime.torch_xla_models.llama import LlamaForCausalLM


class TestYourModule(unittest.TestCase):
  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    torch_xla.manual_seed(42)
    self.vocab_size = 128
    config = AutoConfig.from_pretrained(
      "meta-llama/Meta-Llama-3-8B",
      num_hidden_layers=1,
      num_attention_heads=8,
      hidden_size=8,
      intermediate_size=16,
      vocab_size=self.vocab_size,
    )
    config.flash_attention = False
    # place model on CPU device first
    with torch.device("cpu"):
      self.hf_model = HfLlamaForCausalLM(config)
      self.model = LlamaForCausalLM(config)
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
      llama_output = model_xla(
        input, labels=input, attention_mask=torch.ones_like(input)
      )
      torch_xla.sync()
      self.assertTrue(
        torch.allclose(hf_output.logits, llama_output.logits, atol=1e-6),
        "logits are not equal",
      )
      self.assertTrue(
        torch.allclose(hf_output.loss, llama_output.loss, atol=1e-6),
        "loss is not equal",
      )

  def test_forward_torch_xla_against_native(self):
    input_size = 8
    device = torch.device("cpu")
    input = torch.randint(self.vocab_size, ((2, input_size // 2)))
    llama_output_native = self.model(
      input, labels=input, attention_mask=torch.ones_like(input)
    )

    device = torch_xla.device()
    input = input.to(device)
    model_xla = copy.deepcopy(self.model).to(device)
    torch_xla.sync()

    llama_output = model_xla(input, labels=input, attention_mask=torch.ones_like(input))
    torch_xla.sync()
    self.assertTrue(
      torch.allclose(
        llama_output_native.logits, llama_output.logits.to("cpu"), atol=1e-2
      ),
      "CPU run and XLA run logits are not equal",
    )
    self.assertTrue(
      torch.allclose(llama_output_native.loss, llama_output.loss.to("cpu"), atol=1e-2),
      "CPU run and XLA run loss is not equal",
    )


if __name__ == "__main__":
  unittest.main()
