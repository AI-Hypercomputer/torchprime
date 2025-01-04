import unittest

from transformers import AutoConfig, LlamaForCausalLM as HfLlamaForCausalLM

import torch
import torch_xla

from torchprime.torch_xla_models.llama import LlamaForCausalLM


class TestYourModule(unittest.TestCase):
    def test_forward(self):
        config = AutoConfig.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            num_hidden_layers=1,
            num_attention_heads=8,
            hidden_size=8,
            intermediate_size=16,
            vocab_size=128,
        )
        config.flash_attention = False

        device = torch_xla.device()
        torch.manual_seed(42)
        torch_xla.manual_seed(42)
        with device:
            hf_model = HfLlamaForCausalLM(config)
            model = LlamaForCausalLM(config)
            model.load_state_dict(hf_model.state_dict())
        torch_xla.sync()

        input_sizes = [8, 128, 256]
        for input_size in input_sizes:
            input = torch.randint(128, ((2, input_size // 2))).to(device)
            hf_output = hf_model(input, labels=input, attention_mask=torch.ones_like(input))
            llama_output = model(input, labels=input, attention_mask=torch.ones_like(input))
            torch_xla.sync()
            self.assertTrue(torch.allclose(hf_output.logits, llama_output.logits, atol=1e-6),
                            "logits are not equal")
            self.assertTrue(torch.allclose(hf_output.loss, llama_output.loss, atol=1e-6),
                            "loss is not equal")


if __name__ == '__main__':
    unittest.main()
