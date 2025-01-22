import pytest

from torchprime.mfu.mfu import compute_mfu


# Reference MFU data is taken from MaxText results:
# https://docs.google.com/spreadsheets/d/10r5oziZr9DiBkVn3ngodu_SuvM7yhDp_qUjKMGWBMPE/edit?resourcekey=0-Cn15GPT-vPFX4vriCrXapQ&gid=76378515#gid=76378515
def test_llama2_7b_v5p_mfu():
  config = {
    "_name_or_path": "meta-llama/Llama-2-7b-hf",
    "architectures": ["LlamaForCausalLM"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 4096,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.0.dev0",
    "use_cache": False,
    "vocab_size": 32000,
  }
  result = compute_mfu(
    config,
    batch_size=1024,
    sequence_length=4096,
    step_duration=2.801027417,
    tpu_name="foobar-v5p-512",
  )
  assert result.mfu == pytest.approx(0.5872846948, rel=0.01, abs=0.01)


def test_llama2_70b_v5e_mfu():
  config = {
    "architectures": ["LlamaForCausalLM"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 8192,
    "initializer_range": 0.02,
    "intermediate_size": 28672,
    "max_position_embeddings": 4096,
    "model_type": "llama",
    "num_attention_heads": 64,
    "num_hidden_layers": 80,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.32.0.dev0",
    "use_cache": False,
    "vocab_size": 32000,
  }
  result = compute_mfu(
    config,
    batch_size=512,
    sequence_length=2048,
    step_duration=14.97010803,
    tpu_name="v5e-256-stuff",
  )
  assert result.mfu == pytest.approx(0.5950, rel=0.01, abs=0.01)


def test_llama3_70b_v6e_mfu():
  config = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "hidden_act": "silu",
    "hidden_size": 8192,
    "initializer_range": 0.02,
    "intermediate_size": 28672,
    "max_position_embeddings": 8192,
    "model_type": "llama",
    "num_attention_heads": 64,
    "num_hidden_layers": 80,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.0.dev0",
    "use_cache": False,
    "vocab_size": 128256,
  }
  # Reference data is from https://docs.google.com/spreadsheets/d/187NXHGdfSaOFMeKzegiNK3ZxMRv9-1RHDJx5rNlzz28/edit?resourcekey=0-JuGULB60wSi1QhidrJEDkA&gid=660879005#gid=660879005
  result = compute_mfu(
    config,
    batch_size=128,
    sequence_length=8192,
    step_duration=16.992,
    tpu_name="abc-v6e-128-stuff",
  )
  assert result.mfu == pytest.approx(0.2562, rel=0.01, abs=0.01)


def test_mixtral_8x7b_mfu():
  config = {
    "architectures": ["MixtralForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 32768,
    "model_type": "mixtral",
    "num_attention_heads": 32,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 8,
    "output_router_logits": False,
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000.0,
    "router_aux_loss_coef": 0.02,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.36.0.dev0",
    "use_cache": True,
    "vocab_size": 32000,
  }
  # Reference data is from https://pantheon.corp.google.com/bigquery?ws=!1m7!1m6!12m5!1m3!1sml-workload-benchmarks!2sus-central1!3s08711ba4-a48f-4fa9-acf2-f8c86673287b!2e1
  result = compute_mfu(
    config,
    batch_size=4608,
    sequence_length=4096,
    step_duration=51.876,
    tpu_name="abc-v5p-256-stuff",
  )
  assert result.mfu == pytest.approx(0.51, rel=0.01, abs=0.01)
