from typing import Any

import torch
from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2


def get_llama3_model(torch_dtype: torch.dtype):
  """Returns the Llama3.2 1B model."""
  config = modeling_llama.LlamaConfig(
      attention_bias=False,
      attention_dropout=0.0,
      bos_token_id=128000,
      eos_token_id=128001,
      head_dim=64,
      hidden_act="silu",
      hidden_size=2048,
      initializer_range=0.02,
      intermediate_size=8192,
      max_position_embeddings=131072,
      mlp_bias=False,
      num_attention_heads=32,
      num_hidden_layers=16,
      num_key_value_heads=8,
      rms_norm_eps=1e-05,
      rope_scaling={
          "factor": 32.0,
          "high_freq_factor": 4.0,
          "low_freq_factor": 1.0,
          "original_max_position_embeddings": 8192,
          "rope_type": "llama3",
      },
      rope_theta=500000.0,
      tie_word_embeddings=True,
      use_cache=True,
      vocab_size=128256,
      _attn_implementation="eager",
  )
  model = modeling_llama.LlamaForCausalLM(config).to(torch_dtype)
  return model


def get_qwen2_model(torch_dtype: torch.dtype):
  """Returns the Qwen2 1.7B model."""
  config = modeling_qwen2.Qwen2Config(
      attention_bias=False,
      attention_dropout=0.0,
      bos_token_id=151643,
      eos_token_id=151645,
      head_dim=128,
      hidden_act="silu",
      hidden_size=2048,
      initializer_range=0.02,
      intermediate_size=6144,
      max_position_embeddings=40960,
      max_window_layers=28,
      num_attention_heads=16,
      num_hidden_layers=28,
      num_key_value_heads=8,
      rms_norm_eps=1e-06,
      rope_scaling=None,
      rope_theta=1000000,
      sliding_window=None,
      tie_word_embeddings=True,
      use_cache=True,
      use_sliding_window=False,
      vocab_size=151936,
      _attn_implementation="eager",
  )
  model = modeling_qwen2.Qwen2ForCausalLM(config).to(torch_dtype)
  return model


def get_model(model_name: str, dtype: torch.dtype) -> Any:
  match model_name:
    case "llama3.2-1B":
      return get_llama3_model(dtype)
    case "qwen2-1.7B":
      return get_qwen2_model(dtype)
    case _:
      raise ValueError(f"Unsupported model: {model_name}")