"""Unit test for the DeepSeek V3 model using BaseCausalLM.

This test verifies that a minimal DeepSeek V3 model can be exported and
reloaded without changing its weights.
"""

import pytest
import torch
import torch_xla.core.xla_model as xm
from omegaconf import OmegaConf

from torchprime.torch_xla_models.model.deepseek_v3 import DeepseekForCausalLM
from torchprime.torch_xla_models.model.model_utils import set_default_dtype


@pytest.fixture(scope="module")
def cfg():
  return OmegaConf.create(
    {
      "model_id": "deepseek-v3-mini",
      "model_class": "deepseek_v3.DeepseekForCausalLM",
      "vocab_size": 128,
      "hidden_size": 64,
      "intermediate_size": 256,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "num_key_value_heads": 1,
      "hidden_act": "silu",
      "max_position_embeddings": 64,
      "bos_token_id": 1,
      "eos_token_id": 2,
      "tokenizer_name": "deepseek-ai/DeepSeek-V3",
      "initializer_range": 0.02,
      "rms_norm_eps": 1e-5,
      "attention_dropout": False,
      "attention_bias": False,
      "attention_kernel": "torch",
      "rope_theta": 10000.0,
    }
  )


@pytest.mark.xla
def test_deepseek_model_export_reload_consistency(tmp_path, cfg):
  device = xm.xla_device()
  with set_default_dtype(torch.bfloat16):
    model = DeepseekForCausalLM(cfg).to(device).eval()
  input_ids = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long, device=device)
  attn_mask = torch.ones_like(input_ids).to(device, dtype=torch.bfloat16)

  with torch.no_grad():
    orig_logits = model(input_ids, attn_mask)[0]
    assert orig_logits.shape == (1, 4, cfg.vocab_size)
  xm.mark_step()

  export_dir = tmp_path / "deepseek_mini_export"
  model.export(str(export_dir))

  reloaded_model = DeepseekForCausalLM(cfg)
  reloaded_model.from_pretrained(str(export_dir))
  reloaded_model.to(device).eval()

  with torch.no_grad():
    reload_logits = reloaded_model(input_ids, attn_mask)[0]
  xm.mark_step()

  diff = (orig_logits - reload_logits).abs().max()
  assert diff.item() < 0.005, f"Max diff {diff.item()} too large"
