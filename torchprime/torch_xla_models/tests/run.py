import os
from contextlib import contextmanager

import torch
import torch_xla.core.xla_model as xm
from omegaconf import OmegaConf

from torchprime.torch_xla_models.model.llama.model import (
  LlamaForCausalLM,
)  # Update if your model resolution is different


@contextmanager
def set_default_dtype(dtype):
  # Get the current default dtype
  previous_dtype = torch.get_default_dtype()
  # Set the new default dtype
  torch.set_default_dtype(dtype)
  try:
    yield
  finally:
    # Revert to the original default dtype
    torch.set_default_dtype(previous_dtype)


def main(tmp_path, cfg):
  device = xm.xla_device()
  with set_default_dtype(torch.bfloat16):
    model = LlamaForCausalLM(cfg).to(device).eval()

  print(
    f"model contains {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
  )

  vocab = cfg.vocab_size
  input_ids = torch.randint(0, vocab, (1, 4), dtype=torch.long, device=device)
  attn_mask = torch.ones_like(input_ids).to(device, dtype=torch.bfloat16)

  with torch.no_grad():
    orig_logits = model(input_ids, attn_mask)[0]
    assert orig_logits.shape == (1, 4, vocab)
  xm.mark_step()

  export_dir = os.path.join(tmp_path, "llama_mini_export")
  model.export(export_dir)

  reloaded = LlamaForCausalLM(cfg)
  reloaded.from_pretrained(export_dir)
  reloaded.to(device).eval()

  with torch.no_grad():
    reload_logits = reloaded(input_ids, attn_mask)[0]

  xm.mark_step()

  diff = (orig_logits - reload_logits).abs().max()
  assert diff.item() < 0.005, f"Max diff {diff.item()} too large"


if __name__ == "__main__":
  config_dict = OmegaConf.create(
    {
      "model_id": "llama-mini",
      "model_class": "llama.LlamaForCausalLM",
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
      "tokenizer_name": "meta-llama/Meta-Llama-3-8B",
      "initializer_range": 0.02,
      "rms_norm_eps": 1e-5,
      "attention_dropout": False,
      "attention_bias": False,
      "attention_kernel": "torch",  # for some reason "flash_attention" does not work
      "rope_theta": 10000.0,
    }
  )

  import tempfile

  with tempfile.TemporaryDirectory() as tmp_dir:
    main(tmp_path=tmp_dir, cfg=config_dict)
    print("Test passed: LLaMA mini model export and reload consistency verified.")
