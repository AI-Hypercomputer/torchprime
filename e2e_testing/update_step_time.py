#!/usr/bin/env python3
"""Query BigQuery for E2E test results and compute step time bounds.

This script queries the specified BigQuery table for recent test results,
computes statistical bounds for each benchmark's step times, and exports
the results to a YAML file for use in GitHub Actions.
"""

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import scipy
import yaml
from google.cloud import bigquery
from rich.console import Console
from rich.table import Table

from torchprime.launcher.benchmark_db_util import TORCHPRIME_SOFTWARE_ID


def match_llama3_8b(row):
  config = json.loads(row.configs_framework)
  return (
    row.run_id.startswith("llama-3-8b-")
    and not row.run_id.startswith("llama-3-8b-sft")
    and config["dcn_mesh"]["data"] == 1
    and config["dcn_mesh"]["fsdp"] == 1
    and config["ici_mesh"]["tensor"] == 1
    and ("context" not in config["ici_mesh"] or config["ici_mesh"]["context"] == 1)
    and (
      "pure_modules" not in config["model"] or len(config["model"]["pure_modules"]) == 0
    )
  )


def match_llama3_8b_pure_mlp(row):
  config = json.loads(row.configs_framework)
  return (
    row.run_id.startswith("llama-3-8b-pure-mlp")
    and config["dcn_mesh"]["data"] == 1
    and config["dcn_mesh"]["fsdp"] == 1
    and config["ici_mesh"]["tensor"] == 1
    and (
      "pure_modules" in config["model"]
      and config["model"]["pure_modules"] == ["LlamaMLP", "EinsumLinear"]
    )
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


def match_llama_3_8b_sft(row):
  config = json.loads(row.configs_framework)
  return (
    row.run_id.startswith("llama-3-8b-sft")
    and config["dcn_mesh"]["fsdp"] == 1
    and config["ici_mesh"]["tensor"] == 1
  )


def match_llama_3_8b_ddp_fsdp(row):
  config = json.loads(row.configs_framework)
  return (
    row.run_id.startswith("llama-3-8b-ddp-fsdp")
    and config["dcn_mesh"]["data"] == 2
    and config["ici_mesh"]["fsdp"] == 4
    and config["ici_mesh"]["tensor"] == 1
  )


def match_llama_3_8b_fsdp_cp(row):
  config = json.loads(row.configs_framework)
  return (
    row.run_id.startswith("llama-3-8b-fsdp-cp")
    and ("context" in config["ici_mesh"] and config["ici_mesh"]["context"] == 2)
    and config["ici_mesh"]["fsdp"] == 2
    and config["ici_mesh"]["tensor"] == 1
  )


def match_ds_v3_debug(row):
  config = json.loads(row.configs_framework)
  return (
    row.run_id.startswith("ds-v3-shallow")
    and config["ici_mesh"]["fsdp"] == 4
    and config["ici_mesh"]["tensor"] == 1
  )


BENCHMARKS = {
  "Llama 3.0 8B": match_llama3_8b,
  "Llama 3.0 8B (@assume_pure)": match_llama3_8b_pure_mlp,
  "Llama 3.1 8B (Splash Attention)": match_llama3_1_8b_sa,
  "Llama 3.1 8B (Scan + Offload)": match_llama3_1_8b_scan_offload,
  "Llama 3.0 8B (2D sharding)": match_llama3_8b_2d,
  "Mixtral 8x7B": match_mixtral,
  "Llama 3.0 8B (2 Slice)": match_llama_3_8b_2_slice,
  "Llama 3.0 8B SFT": match_llama_3_8b_sft,
  "Llama 3.0 8B (ddp + fsdp)": match_llama_3_8b_ddp_fsdp,
  "Llama 3.0 8B (fsdp + cp)": match_llama_3_8b_fsdp_cp,
  "Deepseek v3 Debug Model": match_ds_v3_debug,
}

STEP_ID_MAPPING = {
  "Llama 3.0 8B": "llama-3-8b",
  "Llama 3.0 8B (@assume_pure)": "llama-3-8b-pure-mlp",
  "Llama 3.1 8B (Splash Attention)": "llama-3_1-8b-sa",
  "Llama 3.1 8B (Scan + Offload)": "llama-3_1-8b-scan-offload",
  "Llama 3.0 8B (2D sharding)": "llama-3-8b-2d",
  "Mixtral 8x7B": "mixtral-8x7b",
  "Llama 3.0 8B (2 Slice)": "llama-3-8b-2-slice",
  "Llama 3.0 8B SFT": "llama-3-8b-sft",
  "Llama 3.0 8B (ddp + fsdp)": "llama-3-8b-ddp-fsdp",
  "Llama 3.0 8B (fsdp + cp)": "llama-3-8b-fsdp-cp",
  "Deepseek v3 Debug Model": "ds-v3-shallow",
}
"""Mapping from the benchmark name to the ID of the E2E test step used in GitHub Actions."""


def parse_days_ago(days_str: str):
  """Parse a string like '2 days ago' and return a datetime object."""
  match = re.match(r"(\d+)\s+days?\s+ago", days_str.strip())
  if not match or len(match.groups()) != 1:
    raise ValueError(f"Invalid days ago format: {days_str}")
  days = int(match.group(1))
  return datetime.now() - timedelta(days=days)


def render_local_datetime_utc(dt):
  """Render a local datetime object in UTC timezone."""
  local_tz = datetime.now().astimezone().tzinfo
  dt = dt.replace(tzinfo=local_tz)
  dt_utc = dt.astimezone(datetime.utcnow().tzinfo)
  return dt_utc.isoformat()


def parse_datetime(datetime_str):
  """Parse datetime string.

  First tries GoogleSQL format with timezone, then falls back to Python datetime
  parsing and converts to GoogleSQL format.
  """
  # First check if it's already in GoogleSQL format (has timezone)
  if "/" in datetime_str or " UTC" in datetime_str:
    return datetime_str

  # Then check if it's in 'N days ago' format
  if "days ago" in datetime_str:
    return render_local_datetime_utc(parse_days_ago(datetime_str))

  # Try common datetime formats and convert to GoogleSQL format
  formats = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
  ]

  for fmt in formats:
    try:
      dt = datetime.strptime(datetime_str, fmt)
      return render_local_datetime_utc(dt)
    except ValueError:
      continue

  # Return as-is and let BigQuery handle it
  return datetime_str


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
  standard_error = stdev / np.sqrt(count)
  _, upper_bound = scipy.stats.t.interval(
    confidence_level, df, loc=0, scale=standard_error
  )

  return upper_bound


def compute_bounds(step_times, confidence_level):
  """Implements the formula described in e2e_testing/README.md"""
  n = len(step_times)
  assert n > 1, "Not enough step times to compute bounds"

  mean = sum(step_times) / n
  min_time = min(step_times)
  max_time = max(step_times)

  stdev = float(np.std(step_times, ddof=1))
  t_critical = float(calculate_confidence_t_interval(1 - confidence_level, stdev, n))

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


def main():
  """
  Query BigQuery for E2E test results and compute step time bounds.

  This script queries the specified BigQuery table for recent test results,
  computes statistical bounds for each benchmark's step times, and exports
  the results to a YAML file for use in GitHub Actions.
  """
  parser = argparse.ArgumentParser(
    description="Query BigQuery for E2E test results and compute step time bounds.",
    formatter_class=argparse.RawTextHelpFormatter,
  )
  parser.add_argument(
    "--bq-project",
    default="tpu-pytorch",
    help="BigQuery project ID",
  )
  parser.add_argument(
    "--bq-dataset",
    default="benchmark_dataset_test",
    help="BigQuery dataset name",
  )
  parser.add_argument(
    "--bq-table",
    default="torchprime-e2e-tests",
    help="BigQuery table name",
  )
  parser.add_argument(
    "--start-time",
    default=parse_days_ago("5 days ago").strftime("%Y-%m-%d %H:%M:%S"),
    help="Start time for the query in GoogleSQL datetime format (e.g., '2025-05-29 17:52:00 America/Los_Angeles').\n"
    "Can also accept common datetime formats which will be converted.\n"
    "In particular, supports '[N] days ago' format, e.g., '2 days ago'.\n"
    "Defaults to 5 days ago.",
  )
  parser.add_argument(
    "--end-time",
    default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    help="End time for the query in GoogleSQL datetime format (e.g., '2025-06-01 20:00:00 America/Los_Angeles').\n"
    "Can also accept common datetime formats which will be converted.\n"
    "In particular, supports '[N] days ago' format, e.g., '2 days ago'.\n"
    "Defaults to the current time.",
  )
  parser.add_argument(
    "--limit",
    default=1200,
    type=int,
    help="Maximum number of rows to retrieve",
  )
  parser.add_argument(
    "--output",
    default="e2e_testing/step_time_bounds.yaml",
    type=str,
    help="Output YAML file path",
  )
  parser.add_argument(
    "--confidence_level",
    default=99.0,
    type=float,
    help="Confidence level, default is 99%",
  )
  args = parser.parse_args()
  bq_project = args.bq_project
  bq_dataset = args.bq_dataset
  bq_table = args.bq_table
  start_time = args.start_time
  end_time = args.end_time
  limit = args.limit
  output = args.output
  confidence_level = args.confidence_level
  console = Console()
  confidence_level = confidence_level / 100.0

  # Parse datetime inputs
  start_time = parse_datetime(start_time)
  end_time = parse_datetime(end_time)

  console.print("[bold]Querying BigQuery:[/bold]")
  console.print(f"  Project: {bq_project}")
  console.print(f"  Dataset: {bq_dataset}")
  console.print(f"  Table: {bq_table}")
  console.print(f"  Start: {start_time}")
  console.print(f"  End: {end_time}")
  console.print(f"  Limit: {limit}")

  client = bigquery.Client()

  query = f"""
  -- Find the most recent rows based on update_timestamp and sort them by most recent first
  SELECT
    *
  FROM
    `{bq_project}`.`{bq_dataset}`.`{bq_table}`
  WHERE
    software_id = '{TORCHPRIME_SOFTWARE_ID}' AND
    update_timestamp >= TIMESTAMP('{start_time}') AND
    update_timestamp <= TIMESTAMP('{end_time}')
  ORDER BY
    update_timestamp DESC
  LIMIT
    {limit};
  """

  query_job = client.query(query)
  rows = list(query_job.result())

  console.print(f"\n[green]Retrieved {len(rows)} rows[/green]\n")

  # Group rows by benchmark
  step_time_by_benchmark = {}

  for row in rows:
    matched = set()
    for name, match_fn in BENCHMARKS.items():
      if match_fn(row):
        matched.add(name)

    if not matched:
      raise ValueError(f"Run ID {row.run_id} does not match any benchmark")

    if len(matched) > 1:
      raise ValueError(f"Run ID {row.run_id} matches multiple benchmarks: {matched}")

    step_time_by_benchmark.setdefault(matched.pop(), []).append(row.metrics_step_time)

  # Ensure all benchmarks are represented
  step_time_by_benchmark = {
    name: step_time_by_benchmark.get(name, []) for name in BENCHMARKS
  }

  # Display results table
  table = Table(title="Confidence Intervals for Step Time")
  table.add_column("Benchmark", justify="right", style="cyan", no_wrap=True)
  table.add_column("Runs", justify="right", style="magenta")
  table.add_column("Average (sec)", justify="right", style="green")
  table.add_column("Lower Bound", justify="right", style="green")
  table.add_column("Upper Bound", justify="right", style="green")
  table.add_column("Range (ms)", justify="right", style="green")

  benchmarks_data = {}

  for name, step_times in step_time_by_benchmark.items():
    if len(step_times) <= 1:
      console.print(f"\n[red]Not enough data to update bounds for: {name}[/red]\n")
      continue
    lower_bound, upper_bound = compute_bounds(step_times, confidence_level)
    console.print(f"updating compute bounds for: {name}")
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

    job_id = STEP_ID_MAPPING.get(name)
    if job_id:
      benchmarks_data[job_id] = {
        "name": name,
        "step_time_lower_bound": round(lower_bound, 8),
        "step_time_upper_bound": round(upper_bound, 8),
        "confidence_interval": round((upper_bound - lower_bound) / 2, 5),
        "average": round(average, 4),
        "sample_size": len(step_times),
      }
      # Manually add target loss values for llama-3-8b-sft benchmark
      # TODO (https://github.com/AI-Hypercomputer/torchprime/issues/348):
      # preserve non-performance releated values in the file.
      if job_id == "llama-3-8b-sft":
        benchmarks_data[job_id].update({"target_loss": 0.4735, "loss_tolerance": 0.001})

  console.print(table)

  # Write to file
  output_path = Path(output)
  output_path.parent.mkdir(exist_ok=True, parents=True)

  with open(output_path, "w") as f:
    yaml.dump(
      {
        "benchmarks": benchmarks_data,
        "metadata": {
          "query_start": start_time,
          "query_end": end_time,
          "confidence_level": confidence_level,
        },
      },
      f,
      default_flow_style=False,
      sort_keys=False,
    )

  console.print(f"\n[green]Performance bounds exported to {output_path}[/green]")


if __name__ == "__main__":
  main()
