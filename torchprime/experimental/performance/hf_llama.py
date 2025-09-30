from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2
import torch
from typing import Any

import time
import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla


def get_llama3_model(torch_dtype: torch.dtype):
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
      model_cpu = get_llama3_model(dtype)
    case "qwen2-1.7B":
      model_cpu = get_qwen2_model(dtype)
    case _:
      raise ValueError(f"Unsupported model: {model_name}")
  return model_cpu


# --- Configuration ---
USE_TORCH_COMPILE = False
BATCH_SIZE = 1
SEQ_LEN = 128
NUM_RUNS = 10

# It's good practice to define the device first.
device = torch_xla.device()

# Create the model on CPU first
model_cpu = get_model("llama3.2-1B", torch.bfloat16)
config = model_cpu.config
model_cpu.eval()  # Set to evaluation mode

# Move model to the XLA device.
model_tpu = model_cpu.to(device)

# Create dummy input_ids and move to the XLA device.
input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
# Move inputs to the XLA device as well.
input_ids = input_ids.to(device)

if USE_TORCH_COMPILE:
  # To use torch.compile with XLA, you should specify the 'openxla' or 'openxla_eval' backend.
  model_tpu = torch.compile(model_tpu)

# Initial run (warm-up) to trigger XLA compilation
print("Warming up...")
with torch.no_grad():
    output_tpu = model_tpu(input_ids).logits

torch_xla.sync()
# Subsequent runs for measurement
print(f"Starting benchmark for {NUM_RUNS} runs...")
times = []
for i in range(NUM_RUNS):
    start_time = time.perf_counter()
    with torch.no_grad():
      # The model forward pass is intentionally not assigned to a variable
      # to measure only the execution time.
      model_tpu(input_ids)
    torch_xla.sync()
    end_time = time.perf_counter()
    times.append(end_time - start_time)
    print(f"Run {i+1}/{NUM_RUNS}: {(end_time - start_time) * 1000:.2f} ms")
    
    
# Print final performance results
print("\n--- Benchmark Results ---")
print(f"Number of runs: {len(times)}")
print(f"Average latency: {np.mean(times) * 1000:.2f} ms")
print(f"Median latency:  {np.median(times) * 1000:.2f} ms")
print(f"P90 latency:     {np.percentile(times, 90) * 1000:.2f} ms")
print(f"Min latency:     {np.min(times) * 1000:.2f} ms")
print(f"Max latency:     {np.max(times) * 1000:.2f} ms")


# Add this line to wait for the TPU to finish and ensure a clean exit
torch_xla.sync()
print("Script finished and exited cleanly.")