import json
import os

from benchmark_db_writer import bq_writer_utils
from benchmark_db_writer.schema.workload_benchmark_v2 import (
  workload_benchmark_v2_schema,
)

from torchprime.launcher import benchmark_db_util


def _write_summary_to_bq_client(
  summary_data: dict,
  bq_project: str,
  bq_dataset: str,
  bq_table: str,
):
  """
  Uploads the prepared benchmark summary to BigQuery.
  This is the low-level client interaction.
  Args:
      summary_data: A dictionary containing the benchmark summary,
                    typically from prepare_benchmark_summary.
      bq_project: The BigQuery project ID.
      bq_dataset: The BigQuery dataset ID.
      bq_table: The BigQuery table ID.
  """
  print("Attempting to upload benchmark results to BigQuery...")
  client = bq_writer_utils.create_bq_writer_object(
    project=bq_project,
    dataset=bq_dataset,
    table=bq_table,
    dataclass_type=workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema,
  )
  summary_obj = workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema(**summary_data)
  client.write([summary_obj])
  print(
    f"Benchmark results for run_id '{summary_obj.run_id}' successfully uploaded to BigQuery.",
    flush=True,
  )


def _get_env_configs() -> dict:
  """Gathers necessary configurations from environment variables."""
  return {
    "gcs_artifact_dir": os.environ["TORCHPRIME_ARTIFACT_DIR"],
    "cluster": os.environ["TORCHPRIME_CLUSTER"],
    "num_slices": os.environ["TORCHPRIME_NUM_SLICES"],
    "bq_project": os.environ["TORCHPRIME_BQ_PROJECT"],
    "bq_dataset": os.environ["TORCHPRIME_BQ_DATASET"],
    "bq_table": os.environ["TORCHPRIME_BQ_TABLE"],
    "tpu_type": os.environ["TORCHPRIME_TPU_TYPE"],
    "comments": os.environ.get("TORCHPRIME_COMMENTS"),
    "docker_url": os.environ["TORCHPRIME_DOCKER_URL"],
    "update_person_ldap": os.environ["TORCHPRIME_USER"],
    "configs_xla_flags": os.environ.get("LIBTPU_INIT_ARGS", ""),
    "all_env": dict(os.environ),  # For capturing the full environment
  }


def collect_and_upload_benchmark_summary(
  process_returncode: int,
  jobset_name: str,
  mounted_artifact_path_str: str,
):
  """
  Gathers necessary information from environment variables and artifacts,
  prepares the benchmark summary, and uploads it to BigQuery.
  """
  env_vars = _get_env_configs()

  metrics = benchmark_db_util.get_metrics(mounted_artifact_path_str, jobset_name)
  config = benchmark_db_util.get_config(mounted_artifact_path_str, jobset_name)

  model_config = config.get("model", {}) if config else {}
  optimizer_config = config.get("optimizer", {}) if config else {}

  summary_dict = benchmark_db_util.prepare_benchmark_summary(
    process_returncode=process_returncode,
    jobset_name=jobset_name,
    tpu_type=env_vars["tpu_type"],
    # The following are passed as **kwargs to prepare_benchmark_summary
    model_id=model_config.get("model_id"),
    update_person_ldap=env_vars["update_person_ldap"],
    cluster_name=env_vars["cluster"],
    hardware_num_slices=env_vars["num_slices"],
    configs_framework=json.dumps(config) if config else None,
    configs_env=json.dumps(env_vars["all_env"]),
    configs_xla_flags=env_vars["configs_xla_flags"],
    configs_container_version=env_vars["docker_url"],
    logs_artifact_directory=env_vars["gcs_artifact_dir"],
    logs_comments=env_vars["comments"],
    gcs_metrics_bucket=os.path.join(env_vars["gcs_artifact_dir"], jobset_name),
    workload_gbs=config.get("global_batch_size") if config else None,
    workload_precision=config.get("torch_dtype"),
    workload_optimizer=optimizer_config.get("type"),
    workload_sequence_length=config.get("block_size") if config else None,
    **metrics,
  )

  _write_summary_to_bq_client(
    summary_data=summary_dict,
    bq_project=env_vars["bq_project"],
    bq_dataset=env_vars["bq_dataset"],
    bq_table=env_vars["bq_table"],
  )
