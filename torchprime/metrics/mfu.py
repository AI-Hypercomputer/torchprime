"""
Calculate Model FLOPs Utilization (MFU).
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from torchprime.metrics.train_flops import (
  DeepSeekConfig,
  Llama4Config,
  LlamaConfig,
  deepseek_tflops,
  llama3_style_models_tflops,
  llama4_tflops,
)


@dataclass
class MFU:
  model_tflops: float
  """The number of floating point operations in the model, in teraflops."""

  hardware_tflops_per_step: float
  """The theoretical hardware floating point throughput during one training step, in teraflops."""

  per_chip_tflops_per_sec: float
  """The realized floating point throughput during one second in each chip, in teraflops."""

  mfu: float
  """Model FLOPs Utilization. Fraction of hardware FLOPs the model uses, from 0 to 1."""


def compute_mfu(
  config: dict,
  batch_size: int,
  sequence_length: int,
  step_duration: float,
  tpu_name: str,
  num_slices: int = 1,
  gradient_accumulation_steps: int = 1,
  torch_dtype: str = "bfloat16",
) -> MFU:
  """
  Calculate MFU of a training config on some TPU hardware.

  Args:

    config: a dictionary representing a decoded JSON HuggingFace-style model config.

    batch_size: global batch size.

    sequence_length: number of tokens in each training example.

    step_duration: duration of one trthroughput_per_deviceaining step.

    tpu_name: accelerator type (e.g. `v5p-128`).

    gradient_accumulation_steps: how many dataloader iterations per optimizer iteration. See \
      https://huggingface.co/docs/accelerate/v0.11.0/en/gradient_accumulation. Defaults to 1.

    torch_dtype: data type used for training (e.g. `bfloat16`).
  """
  model_id = config.get("model_id", "")
  hidden_size = int(config["hidden_size"])

  if "deepseek" in model_id:
    cfg = DeepSeekConfig(
      per_device_batch_size=batch_size,
      max_target_length=sequence_length,
      num_decoder_layers=int(config["num_hidden_layers"]),
      emb_dim=hidden_size,
      num_query_heads=int(config["num_attention_heads"]),
      num_kv_heads=int(config["num_key_value_heads"]),
      head_dim=hidden_size // int(config["num_attention_heads"]),
      mlp_dim=int(config["intermediate_size"]),
      vocab_size=int(config["vocab_size"]),
      mlp_activations=("silu", "linear"),
      gradient_accumulation_steps=gradient_accumulation_steps,
      moe_mlp_dim=int(config["moe_intermediate_size"]),
      num_experts=int(config["n_routed_experts"]),
      num_experts_per_tok=int(config["num_experts_per_tok"]),
      shared_experts=int(config["n_shared_experts"]),
      first_num_dense_layers=int(config["first_k_dense_replace"]),
      qk_nope_head_dim=int(config["qk_nope_head_dim"]),
      qk_rope_head_dim=int(config["qk_rope_head_dim"]),
      v_head_dim=int(config["v_head_dim"]),
      q_lora_rank=int(config.get("q_lora_rank", 0)),
      kv_lora_rank=int(config["kv_lora_rank"]),
    )
    total_tflops, _, _ = deepseek_tflops(cfg)
  elif "llama-4" in model_id or int(config.get("num_local_experts", 1)) > 1:
    cfg = Llama4Config(
      per_device_batch_size=batch_size,
      max_target_length=sequence_length,
      num_decoder_layers=int(config["num_hidden_layers"]),
      emb_dim=hidden_size,
      num_query_heads=int(config["num_attention_heads"]),
      num_kv_heads=int(config["num_key_value_heads"]),
      head_dim=hidden_size // int(config["num_attention_heads"]),
      mlp_dim=int(config["intermediate_size"]),
      vocab_size=int(config["vocab_size"]),
      mlp_activations=("silu", "linear"),
      gradient_accumulation_steps=gradient_accumulation_steps,
      moe_mlp_dim=int(config.get("moe_intermediate_size", config["intermediate_size"])),
      num_experts=int(config.get("num_local_experts", 1)),
      num_experts_per_tok=int(config.get("num_experts_per_tok", 1)),
      shared_experts=int(
        config.get("n_shared_experts", config.get("shared_experts", 0))
      ),
      interleave_moe_layer_step=int(config.get("interleave_moe_layer_step", 1)),
      chunk_attn_window_size=int(config.get("chunk_attn_window_size", sequence_length)),
      nope_layer_interval=int(config.get("nope_layer_interval", 1)),
    )
    total_tflops, _, _ = llama4_tflops(cfg)
  else:
    cfg = LlamaConfig(
      per_device_batch_size=batch_size,
      max_target_length=sequence_length,
      num_decoder_layers=int(config["num_hidden_layers"]),
      emb_dim=hidden_size,
      num_query_heads=int(config["num_attention_heads"]),
      num_kv_heads=int(config["num_key_value_heads"]),
      head_dim=hidden_size // int(config["num_attention_heads"]),
      mlp_dim=int(config["intermediate_size"]),
      vocab_size=int(config["vocab_size"]),
      mlp_activations=("silu", "linear"),
      gradient_accumulation_steps=gradient_accumulation_steps,
    )
    total_tflops, _, _ = llama3_style_models_tflops(cfg)

  # print step_duration after flops calculation
  print(f"step duration: {step_duration}s")
  
  assert torch_dtype == "bfloat16", f"Unsupported dtype {torch_dtype}"

  chip_count_per_slice, tflops_per_chip = get_num_chips_and_tflops_per_chip(tpu_name)

  chip_count = chip_count_per_slice * num_slices
  hw_tflops = step_duration * chip_count * tflops_per_chip
  return MFU(
    model_tflops=total_tflops,
    hardware_tflops_per_step=hw_tflops,
    per_chip_tflops_per_sec=total_tflops / step_duration / chip_count,
    mfu=total_tflops / hw_tflops,
  )


def get_num_chips_and_tflops_per_chip(tpu_name: str) -> tuple[int, int]:
  """
  Determines the number of chips and TFLOPs per chip for a given TPU type.

  Args:
    tpu_name: The name of the TPU (e.g., "v4-8", "v5p-256").

  Returns:
    A tuple containing:
      - chip_count (int): The number of physical TPU chips.
      - tflops_per_chip (int): The peak TFLOPs (BF16) per chip.
  """
  version, core_count = parse_tpu_name(tpu_name)
  match version:
    case "v4":
      # See https://cloud.google.com/tpu/docs/v4
      chip_count = core_count / 2
      tflops_per_chip = 275
    case "v5p":
      # See https://cloud.google.com/tpu/docs/v5p
      chip_count = core_count / 2
      tflops_per_chip = 459
    case "v5e":
      # See https://cloud.google.com/tpu/docs/v5e
      chip_count = core_count
      tflops_per_chip = 197
    case "v6e":
      # See https://cloud.google.com/tpu/docs/v6e
      chip_count = core_count
      tflops_per_chip = 918
    case _:
      raise ValueError(f"Unsupported accelerator type {tpu_name}")
  return chip_count, tflops_per_chip


def parse_tpu_name(s) -> tuple[str, int]:
  match = re.search(r"(v..)-(\d+)", s)
  if match:
    return match.group(1), int(match.group(2))
  match = re.search(r"(v4)-(\d+)", s)
  if match:
    return match.group(1), int(match.group(2))
  raise ValueError(f"No matching pattern found in {s}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Calculate MFU (CLI example).")

  parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Model config path (it should be a HuggingFace-style JSON file)",
  )
  parser.add_argument("--batch-size", type=int, required=True, help="Size of the batch")
  parser.add_argument(
    "--step-duration", type=float, required=True, help="Duration of one step in seconds"
  )
  parser.add_argument(
    "--seq-len",
    type=float,
    required=True,
    help="Number of tokens in each training example",
  )
  parser.add_argument(
    "--tpu-name", type=str, required=True, help="Name of the TPU (e.g. v5p-128)"
  )

  args = parser.parse_args()

  global_batch_size = args.batch_size
  seq_len = args.seq_len
  step_duration = args.step_duration

  mfu = compute_mfu(
    config=json.loads((Path(args.config)).read_text()),
    batch_size=global_batch_size,
    sequence_length=seq_len,
    step_duration=step_duration,
    tpu_name=args.tpu_name,
  )

  print(f"Model teraflops: {mfu.model_tflops}", file=sys.stderr)
  print(f"Hardware teraflops: {mfu.hardware_tflops_per_step}", file=sys.stderr)

  print(mfu.mfu)
