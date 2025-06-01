#!/usr/bin/env python3
"""
This script is used to query the `torchprime-e2e-tests` table in the
`benchmark_dataset_test` dataset of the `tpu-pytorch` project. It retrieves the
most recent rows based on the `update_timestamp` field, filtering for entries
related to the `pytorch_torchprime` software ID within a specific date range.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
import yaml
from google.cloud import bigquery
from rich.console import Console
from rich.table import Table

from torchprime.launcher.benchmark_db_util import TORCHPRIME_SOFTWARE_ID

client = bigquery.Client()

project = "tpu-pytorch"
dataset = "benchmark_dataset_test"
table = "torchprime-e2e-tests"
start_time = "2025-05-29 17:52:00 America/Los_Angeles"
end_time = "2025-06-02 17:52:00 America/Los_Angeles"
limit = 750

QUERY = f"""
-- Find the most recent rows based on update_timestamp and sort them by most recent first
SELECT
  *
FROM
  `{project}`.`{dataset}`.`{table}`
WHERE
  software_id = '{TORCHPRIME_SOFTWARE_ID}' AND
  update_timestamp >= TIMESTAMP('{start_time}') AND
  update_timestamp <= TIMESTAMP('{end_time}')
ORDER BY
  update_timestamp DESC
LIMIT
  {limit};
"""
query_job = client.query(QUERY)
rows = list(query_job.result())


def match_llama3_8b(row):
  config = json.loads(row.configs_framework)
  return (
    row.run_id.startswith("llama-3-8b-")
    and config["dcn_mesh"]["fsdp"] == 1
    and config["ici_mesh"]["tensor"] == 1
  )


def match_llama3_1_8b_sa(row):
  config = json.loads(row.configs_framework)
  return (
    row.run_id.startswith("llama-3dot1-8b-sa")
    and config["model"]["attention_kernel"] == "splash_attention"
  )


def match_llama3_1_8b_scan_offload(row):
  config = json.loads(row.configs_framework)
  return (
    row.run_id.startswith("llama-3dot1-8b-")
    and config["model"]["remat"]["scan_layers"] == "model.layers"
    and config["dcn_mesh"]["fsdp"] == 1
    and config["ici_mesh"]["tensor"] == 1
  )


def match_llama3_8b_2d(row):
  config = json.loads(row.configs_framework)
  return (
    row.run_id.startswith("llama-3-8b-2d")
    and config["dcn_mesh"]["fsdp"] == 1
    and config["ici_mesh"]["fsdp"] == 2
    and config["ici_mesh"]["tensor"] == 2
  )


def match_mixtral(row):
  config = json.loads(row.configs_framework)
  return row.run_id.startswith("mixtral-8x7b-") and config["ici_mesh"]["fsdp"] == 4


def match_llama_3_8b_2_slice(row):
  config = json.loads(row.configs_framework)
  return (
    row.run_id.startswith("llama-3-8b-2-slice")
    and config["dcn_mesh"]["fsdp"] == 2
    and config["ici_mesh"]["fsdp"] == 4
  )


BENCHMARKS = {
  "Llama 3.0 8B": match_llama3_8b,
  "Llama 3.1 8B (Splash Attention)": match_llama3_1_8b_sa,
  "Llama 3.1 8B (Scan + Offload)": match_llama3_1_8b_scan_offload,
  "Llama 3.0 8B (2D sharding)": match_llama3_8b_2d,
  "Mixtral 8x7B": match_mixtral,
  "Llama 3.0 8B (2 Slice)": match_llama_3_8b_2_slice,
}
CONFIDENCE_LEVEL = 0.999  # 99.9% confidence level


def calculate_confidence_t_interval(alpha, stdev, count):
  """Calculate margin of error using t-distribution."""

  if count <= 1 or stdev < 0:
    raise ValueError(
      f"Invalid parameters for t-distribution: count={count}, stdev={stdev}"
    )
  if stdev == 0:
    return 0.0

  df = count - 1
  confidence_level = 1 - alpha
  sem = stdev / np.sqrt(count)
  _, upper_bound = scipy.stats.t.interval(confidence_level, df, loc=0, scale=sem)

  return upper_bound


def compute_bounds(step_times, confidence_level=CONFIDENCE_LEVEL):
  """Implements the formula described in e2e_testing/README.md"""

  n = len(step_times)
  assert n > 1, "Not enough step times to compute bounds"

  mean = sum(step_times) / n
  min_time = min(step_times)
  max_time = max(step_times)

  # Use sample standard deviation for consistency with t-distribution
  stdev = (sum((x - mean) ** 2 for x in step_times) / (n - 1)) ** 0.5
  t_critical = calculate_confidence_t_interval(1 - confidence_level, stdev, n)

  # Calculate the half-width H
  H = max(
    t_critical,
    0.015 * mean,
    max_time - mean,
    mean - min_time,
  )

  lower_bound: float = max(0, mean - H)
  upper_bound: float = mean + H

  return lower_bound, upper_bound


step_time_by_benchmark = {}

for row in rows:
  matched = set()
  for name, match_fn in BENCHMARKS.items():
    if match_fn(row):
      matched.add(name)
  if not matched:
    raise ValueError(f"Run ID {row.run_id} does not match any benchmark: {matched}")
  if len(matched) > 1:
    raise ValueError(f"Run ID {row.run_id} matches multiple benchmarks: {matched}")
  step_time_by_benchmark.setdefault(matched.pop(), []).append(row.metrics_step_time)

step_time_by_benchmark = {name: step_time_by_benchmark[name] for name in BENCHMARKS}


console = Console()
table = Table(title="Confidence Intervals for Step Time")

table.add_column("Benchmark", justify="right", style="cyan", no_wrap=True)
table.add_column("Runs", justify="right", style="magenta")
table.add_column("Average (sec)", justify="right", style="green")
table.add_column("Lower Bound", justify="right", style="green")
table.add_column("Upper Bound", justify="right", style="green")
table.add_column("Range (ms)", justify="right", style="green")

for name, step_times in step_time_by_benchmark.items():
  lower_bound, upper_bound = compute_bounds(step_times)
  average = sum(step_times) / len(step_times)
  interval_ms = (upper_bound - lower_bound) * 1000
  table.add_row(
    name,
    f"{len(step_times)}",
    f"{average:.2f}",
    f"{lower_bound:.4f}",
    f"{upper_bound:.4f}",
    f"{interval_ms:.1f}",
  )

console.print(table)

benchmarks_data = {}
for name, step_times in step_time_by_benchmark.items():
  if len(step_times) <= 1:
    console.print(
      f"[yellow]Warning: Skipping {name} - insufficient data points[/yellow]"
    )
    continue

  lower_bound, upper_bound = compute_bounds(step_times)
  average = sum(step_times) / len(step_times)

  # Map benchmark names to their GitHub Action job IDs
  job_id_mapping = {
    "Llama 3.0 8B": "llama-3-8b",
    "Llama 3.1 8B (Splash Attention)": "llama-3_1-8b-sa",
    "Llama 3.1 8B (Scan + Offload)": "llama-3_1-8b-scan-offload",
    "Llama 3.0 8B (2D sharding)": "llama-3-8b-2d",
    "Mixtral 8x7B": "mixtral-8x7b",
    "Llama 3.0 8B (2 Slice)": "llama-3-8b-2-slice",
  }

  job_id = job_id_mapping.get(name)
  if job_id:
    benchmarks_data[job_id] = {
      "name": name,
      "step_time_lower_bound": round(lower_bound, 8),
      "step_time_upper_bound": round(upper_bound, 8),
      "confidence_interval": round((upper_bound - lower_bound) / 2, 5),
      "average": round(average, 4),
      "sample_size": len(step_times),
    }

# Write to file
output_path = Path("e2e_testing/step_time_bounds.yaml")
output_path.parent.mkdir(exist_ok=True)

with open(output_path, "w") as f:
  yaml.dump(
    {
      "benchmarks": benchmarks_data,
      "metadata": {
        "query_start": start_time,
        "query_end": end_time,
        "confidence_level": CONFIDENCE_LEVEL,
      },
    },
    f,
    default_flow_style=False,
    sort_keys=False,
  )

console.print(f"\nPerformance bounds exported to {output_path}")
