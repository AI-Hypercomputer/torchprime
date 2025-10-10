import argparse
import os
import time
from typing import Any

import numpy as np
import torch
import torch_xla

from torchprime.experimental.benchmark.hf_model import get_model


def main(args):
  # --- Configuration ---
  print("Running in PyTorch/XLA experimental eager mode.")
  torch_xla.experimental.eager_mode(True)

  dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
  torch_dtype = dtype_map[args.dtype]

  # It's good practice to define the device first.
  device = torch_xla.device()

  # Create the model on CPU first
  model_cpu = get_model(args.model_name, torch_dtype)
  config = model_cpu.config
  model_cpu.eval()  # Set to evaluation mode

  # Move model to the XLA device.
  model_tpu = model_cpu.to(device)

  # Create dummy input_ids and move to the XLA device.
  input_ids = torch.randint(
      0, config.vocab_size, (args.batch_size, args.seq_len), dtype=torch.long
  )
  # Move inputs to the XLA device as well.
  input_ids = input_ids.to(device)

  # Preheat the cache.
  print("Preheating...")
  preheat_start_time = time.perf_counter()
  with torch.no_grad():
    _ = model_tpu(input_ids).logits
  preheat_end_time = time.perf_counter()
  preheat_time = preheat_end_time - preheat_start_time
  print(f"PREHEAT WALL TIME: {preheat_time*1000:.4f} ms")

  # Initial run (warm-up)
  print("Warming up...")
  warmup_start_time = time.perf_counter()
  with torch.no_grad():
    _ = model_tpu(input_ids).logits
  warmup_end_time = time.perf_counter()
  warmup_time = warmup_end_time - warmup_start_time

  # Subsequent runs for measurement
  print(f"Starting benchmark for {args.num_runs} runs...")
  times = []
  for i in range(args.num_runs):
    start_time = time.perf_counter()
    with torch.no_grad():
      # The model forward pass is intentionally not assigned to a variable
      # to measure only the execution time.
      model_tpu(input_ids)
      
    # Do we need this???
    torch_xla.sync()

    end_time = time.perf_counter()
    times.append(end_time - start_time)
    print(f"Run {i+1}/{args.num_runs}: {(end_time - start_time) * 1000:.2f} ms")

  # Print final performance results
  print("\n--- Benchmark Results (Eager Mode) ---")
  print(f"Model: {args.model_name}, DType: {args.dtype}")
  print(f"Batch Size: {args.batch_size}, Sequence Length: {args.seq_len}")
  print(f"Preheat time:    {preheat_time * 1000:.2f} ms")
  print(f"Warm-up time:    {warmup_time * 1000:.2f} ms")
  print(f"Number of runs: {len(times)}")
  print(f"Average latency: {np.mean(times) * 1000:.2f} ms")
  print(f"Median latency:  {np.median(times) * 1000:.2f} ms")
  print(f"P90 latency:     {np.percentile(times, 90) * 1000:.2f} ms")
  print(f"Min latency:     {np.min(times) * 1000:.2f} ms")
  print(f"Max latency:     {np.max(times) * 1000:.2f} ms")

  print("Script finished and exited cleanly.")
  os._exit(0)  # <-- Use os._exit() instead of sys.exit()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Benchmark HF models on XLA (Eager Mode).")
  parser.add_argument(
      "--model_name",
      type=str,
      default="llama3.2-1B",
      choices=["llama3.2-1B", "qwen2-1.7B"],
      help="Model to benchmark (must match a config file name).",
  )
  parser.add_argument(
      "--dtype",
      type=str,
      default="bfloat16",
      choices=["bfloat16", "float32"],
      help="Data type for the model.",
  )
  parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
  parser.add_argument("--seq_len", type=int, default=128, help="Sequence length.")
  parser.add_argument("--num_runs", type=int, default=10, help="Number of benchmark runs.")
  main(parser.parse_args())