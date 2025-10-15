import argparse
import os
import time
from typing import Any

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm

from torchprime.experimental.benchmark.hf_model import get_model


def main(args):
  # --- Configuration ---
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

  # Initial run (warm-up) to trigger XLA compilation
  print("Warming up (includes XLA graph compilation)...")
  warmup_start_time = time.perf_counter()
  with torch.no_grad():
    # The first run triggers compilation, which is a one-time cost.
    # Subsequent runs will be much faster as they hit the compilation cache.
    logits = model_tpu(input_ids).logits
  xm.wait_device_ops()  # Block until the graph compilation and execution is complete.
  warmup_end_time = time.perf_counter()
  warmup_time = warmup_end_time - warmup_start_time

  # Subsequent runs for measurement
  print(f"Starting benchmark for {args.num_runs} runs...")
  times = []
  for i in range(args.num_runs):
    start_time = time.perf_counter()
    with torch.no_grad():
      # Assign to a variable to prevent garbage collection before sync.
      logits = model_tpu(input_ids).logits

    xm.wait_device_ops()  # Block until the step's computation is complete for accurate timing.
    end_time = time.perf_counter()
    times.append(end_time - start_time)
    print(f"Run {i+1}/{args.num_runs}: {(end_time - start_time) * 1000:.2f} ms")

  # Print final performance results
  print("\n--- Benchmark Results (Lazy Mode) ---")
  print(f"Model: {args.model_name}, DType: {args.dtype}")
  print(f"Batch Size: {args.batch_size}, Sequence Length: {args.seq_len}")
  print(f"Warm-up time:    {warmup_time * 1000:.2f} ms (includes compilation)")
  print(f"Number of runs: {len(times)}")
  print(f"Average latency: {np.mean(times) * 1000:.2f} ms")
  print(f"Median latency:  {np.median(times) * 1000:.2f} ms")
  print(f"P90 latency:     {np.percentile(times, 90) * 1000:.2f} ms")
  print(f"Min latency:     {np.min(times) * 1000:.2f} ms")
  print(f"Max latency:     {np.max(times) * 1000:.2f} ms")

  # Add this line to wait for the TPU to finish and ensure a clean exit
  xm.wait_device_ops()  # Final sync to ensure all pending operations are done.
  print("Script finished and exited cleanly.")
  os._exit(0)  # <-- Use os._exit() instead of sys.exit()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Benchmark HF models on XLA (Lazy Mode).")
  parser.add_argument(
      "--model_name",
      type=str,
      default="llama3.2-1B",
      choices=["llama3.2-1B", "qwen3-1.7B"],
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