"""
tp is a CLI for common torchprime workflows.
"""

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import toml
from dataclasses_json import dataclass_json
from huggingface_hub.errors import RepositoryNotFoundError
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern  # type: ignore
from rich.text import Text
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import torchprime.launcher.doctor
from torchprime.launcher import save_hf_tokenizer_and_model
from torchprime.launcher.buildpush import buildpush
from torchprime.launcher.util import run_docker

_DOCKER_ENV_FORWARD_LIST = [
  "HF_TOKEN",
  "XLA_IR_DEBUG",
  "XLA_HLO_DEBUG",
  "LIBTPU_INIT_ARGS",
]


@dataclass_json
@dataclass
class Config:
  cluster: str
  project: str
  zone: str
  num_slices: int
  tpu_type: str
  artifact_dir: str
  upload_metrics: bool | None = False
  bq_project: str | None = None
  bq_dataset: str | None = None
  bq_table: str | None = None
  docker_project: str | None = None


def save_hf_model_files_to_gcs(
  repo_id: str, gcs_path: str, file_type: str, temp_dir: str | None
):
  """Downloads model and tokenizer files from Hugging Face Hub and saves them to Google Cloud Storage."""
  print(f"Preparing to save '{file_type}' files from '{repo_id}' to '{gcs_path}'...")
  try:
    save_hf_tokenizer_and_model.save_hf_model_files_to_gcs(
      repo_id, gcs_path, file_type=file_type, temp_dir=temp_dir
    )
    print(f"  -> Successfully saved files to {gcs_path}")
  except RepositoryNotFoundError:
    print(f"\n❌ Error: Repository '{repo_id}' not found.")
    print("Please check the following:")
    print(f"1. The repository ID '{repo_id}' is spelled correctly.")
    print(
      "2. If it's a gated repository, ensure you are authenticated by running 'huggingface-cli login' or exporting your HF_TOKEN."
    )
  except Exception as e:
    print(f"\n❌ An unexpected error occurred for repository '{repo_id}': {e}")


def use(
  cluster: str,
  project: str,
  zone: str,
  num_slices: int,
  tpu_type: str,
  artifact_dir: str,
  upload_metrics: bool,
  bq_project: str,
  bq_dataset: str,
  bq_table: str,
  docker_project: str | None,
):
  """
  Sets up various config like XPK cluster name, GCP project, etc for all
  subsequent commands to use. Typically, you would only run this command once when
  you first clone the repo, or when switching to a different hardware/cluster.

  This will also create and activate a gcloud configuration so that you don't
  have to type the project and zone if you drop down to xpk.
  """
  config = Config(
    cluster=cluster,
    project=project,
    zone=zone,
    num_slices=num_slices,
    tpu_type=tpu_type,
    artifact_dir=artifact_dir,
    upload_metrics=upload_metrics,
    bq_project=bq_project,
    bq_dataset=bq_dataset,
    bq_table=bq_table,
    docker_project=docker_project,
  )
  gcloud_config_name = f"torchprime-{project}-{zone}"
  create_and_activate_gcloud(gcloud_config_name, config)
  assert artifact_dir.startswith("gs://"), (
    f"{artifact_dir} must be in a GCS bucket (start with gs://)"
  )

  path = write_config(config)
  print(f"Written config {path.relative_to(os.getcwd())}")
  torchprime.launcher.doctor.check_all(config)


def create_and_activate_gcloud(gcloud_config_name, config: Config):
  print("Activating gcloud config...")
  ensure_command("gcloud")
  all_configurations = json.loads(
    subprocess.check_output(
      ["gcloud", "config", "configurations", "list", "--format", "json"]
    )
  )
  assert isinstance(all_configurations, list)
  existing = False
  for gcloud_config in all_configurations:
    if gcloud_config["name"] == gcloud_config_name:
      existing = True
      break
  runner = CommandRunner()
  if existing:
    runner.run(
      [
        "gcloud",
        "config",
        "configurations",
        "activate",
        gcloud_config_name,
      ],
    )
  else:
    runner.run(
      [
        "gcloud",
        "config",
        "configurations",
        "create",
        gcloud_config_name,
        "--activate",
      ],
    )

  runner.run(
    [
      "gcloud",
      "config",
      "set",
      "billing/quota_project",
      config.project,
    ],
  )
  runner.run(
    [
      "gcloud",
      "config",
      "set",
      "compute/zone",
      config.zone,
    ],
  )
  runner.run(
    [
      "gcloud",
      "config",
      "set",
      "project",
      config.project,
    ],
  )


def docker_run(use_hf: bool, args: list[str]):
  """
  Runs the provided training command locally for quick testing.
  """
  print(get_project_dir().absolute())

  # Build docker image.
  build_arg = ["USE_TRANSFORMERS=true"] if use_hf else None
  placeholder_url = "torchprime-dev:local"
  docker_url = buildpush(
    push_docker=False, placeholder_url=placeholder_url, build_arg=build_arg
  )
  # Forward a bunch of important env vars.
  env_forwarding = [
    arg for env_var in _DOCKER_ENV_FORWARD_LIST for arg in forward_env(env_var)
  ]
  args = list(v for v in args if v != "")
  command = [
    "python",
  ] + list(args)
  docker_command = [
    "run",
    "-i",
    *env_forwarding,
    "--privileged",
    "--net",
    "host",
    "--shm-size=16G",
    "--rm",
    "-v",
    f"{os.getcwd()}:/workspace",
    "-w",
    "/workspace",
    docker_url,
  ] + command
  run_docker(docker_command)


def run(
  name: str | None,
  base_docker_url: str | None,
  num_slices: int | None,
  priority: str | None,
  use_hf: bool,
  use_local_wheel: bool,
  comments: str | None,
  args: list[str],
):
  """
  Runs the provided SPMD training command as an xpk job on a GKE cluster.
  """
  config = read_config()

  print(get_project_dir().absolute())

  # Build docker image.
  build_arg = []
  if use_hf:
    build_arg.append("USE_TRANSFORMERS=true")
  if use_local_wheel:
    build_arg.append("USE_LOCAL_WHEEL=true")
  docker_project = config.docker_project
  if docker_project is None:
    docker_project = config.project
  docker_url = buildpush(
    torchprime_project_id=docker_project,
    build_arg=build_arg,
    base_docker_url=base_docker_url,
  )

  # Submit xpk workload
  workload_name = name
  if workload_name is None:
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    workload_name = (
      f"{os.environ['USER']}-xpk-{config.tpu_type}-{config.num_slices}-{datetime_str}"
    )

  if not (
    re.match(r"[a-z]([-a-z0-9]*[a-z0-9])?", workload_name) and len(workload_name) < 40
  ):
    raise RuntimeError(
      f"""
      Workload name: {workload_name} not valid. Workload name must match
      [a-z]([-a-z0-9]*[a-z0-9])? and be less than 40 characters long. Consider
      using "--name" flag to set correct name
      """
    )

  command = ["python", "torchprime/launcher/thunk.py"] + list(args)

  if num_slices is None:
    num_slices = config.num_slices

  # Forward a bunch of important env vars.
  env_forwarding = [
    arg for env_var in _DOCKER_ENV_FORWARD_LIST for arg in forward_env(env_var)
  ]
  # Pass configuration, jobset name, and current user as env vars.
  artifact_arg = [
    "--env",
    f"TORCHPRIME_ARTIFACT_DIR={config.artifact_dir}",
    "--env",
    f"TORCHPRIME_TPU_TYPE={config.tpu_type}",
    "--env",
    f"TORCHPRIME_NUM_SLICES={num_slices}",
    "--env",
    f"TORCHPRIME_CLUSTER={config.cluster}",
    "--env",
    f"TORCHPRIME_JOBSET_NAME={workload_name}",
    "--env",
    f"TORCHPRIME_COMMENTS={comments}",
    "--env",
    f"TORCHPRIME_DOCKER_URL={docker_url}",
    "--env",
    f"TORCHPRIME_USER={getpass.getuser()}",
  ]

  if config.upload_metrics:
    artifact_arg.extend(
      [
        "--env",
        f"TORCHPRIME_UPLOAD_METRICS={config.upload_metrics}",
        "--env",
        f"TORCHPRIME_BQ_PROJECT={config.bq_project}",
        "--env",
        f"TORCHPRIME_BQ_DATASET={config.bq_dataset}",
        "--env",
        f"TORCHPRIME_BQ_TABLE={config.bq_table}",
      ]
    )

  ensure_command("xpk")
  xpk_command = (
    [
      "xpk",
      "workload",
      "create",
      "--cluster",
      config.cluster,
      "--docker-image",
      docker_url,
      "--workload",
      workload_name,
      "--tpu-type",
      config.tpu_type,
      "--num-slices",
      str(num_slices),
      "--priority",
      priority if priority else "medium",
      "--zone",
      config.zone,
      "--project",
      config.project,
      "--enable-debug-logs",
      # The following lets xpk propagate user program failures as jobset exit code.
      "--restart-on-user-code-failure",
      "--max-restarts",
      "0",
    ]
    + env_forwarding
    + artifact_arg
    + ["--command", " ".join(command)]
  )
  subprocess.run(xpk_command, check=True)

  styled_workload = Text(workload_name, style="bold green")
  styled_cluster = Text(config.cluster, style="bold green")
  styled_artifacts = Text(f"{config.artifact_dir}/{workload_name}", style="bold green")
  print(f"""
Workload {styled_workload} submitted to cluster {styled_cluster}

Artifacts are stored at {styled_artifacts}
""")


def test(args: list[str]):
  """
  Runs unit tests in torchprime by forwarding arguments to pytest.
  """
  ensure_command("pytest")
  try:
    subprocess.run(["pytest"] + list(args), check=True)
  except subprocess.CalledProcessError as e:
    sys.exit(e.returncode)


def doctor():
  """
  Checks for any problems in your environment (missing packages, credentials, etc.).
  """
  torchprime.launcher.doctor.check_all(config=None)


class CommandRunner:
  def __init__(self):
    self.outputs = b""

  def run(self, command, **kwargs):
    try:
      self.outputs += f">> {' '.join(command)}\n".encode()
      self.outputs += subprocess.check_output(
        command, **kwargs, stderr=subprocess.STDOUT
      )
      self.outputs += b"\n"
    except subprocess.CalledProcessError as e:
      print("Previous command outputs:")
      print(self.outputs.decode("utf-8"))
      print()
      print(f"❌ Error running `{' '.join(command)}` ❌")
      print()
      print(e.stdout)
      sys.exit(-1)


def forward_env(name: str) -> list[str]:
  if name in os.environ:
    return ["--env", f"{name}={os.environ[name]}"]
  return []


def get_project_dir() -> Path:
  script_dir = Path(__file__).parent
  return script_dir.parent.parent.absolute()


def get_config_dir() -> Path:
  project_dir = get_project_dir()
  return project_dir.joinpath(".config")


DEFAULT_CONFIG_NAME = "default.toml"


def write_config(config: Config):
  config_dir = get_config_dir()
  config_dir.mkdir(exist_ok=True)
  default_config = config_dir / DEFAULT_CONFIG_NAME
  default_config.write_text(toml.dumps(config.to_dict()))  # type:ignore
  return default_config


def read_config() -> Config:
  config_path = get_config_dir() / DEFAULT_CONFIG_NAME
  if not config_path.exists():
    raise RuntimeError(f"No config found at {config_path}. Run `tp use` first.")
  return Config.from_dict(toml.load(config_path))  # type:ignore


def ensure_command(name: str):
  """Checks that the `name` program is installed."""
  try:
    subprocess.check_call(
      ["which", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
  except subprocess.CalledProcessError as err:
    raise RuntimeError(
      f"Command `{name}` not found. Make sure it is installed."
    ) from err


class FileChangeHandler(FileSystemEventHandler):
  def __init__(self, command_context, gitignore_spec):
    self.command_context = command_context
    self.gitignore_spec = gitignore_spec
    self.last_trigger_time = time.time()
    self.last_modified_file = ""
    self.file_modified = threading.Condition()
    self.run_command_thread = threading.Thread(target=self.run_command_thread_fn)
    self.run_command_thread.daemon = True
    self.run_command_thread.start()

    # Trigger initial run
    with self.file_modified:
      self.file_modified.notify()

  def on_modified(self, event):
    if event.is_directory:
      return

    # Check if file matches gitignore patterns
    relative_path = os.path.relpath(str(event.src_path), str(get_project_dir()))
    if self.gitignore_spec.match_file(relative_path):
      return

    # Exclude `.git` directory
    if ".git" in relative_path.split(os.sep):
      return

    # Debounce frequent modifications.
    current_time = time.time()
    if current_time - self.last_trigger_time < 1.0:
      return
    self.last_trigger_time = current_time

    # Raise a condition variable to signal that the file has been modified.
    with self.file_modified:
      self.last_modified_file = str(event.src_path)
      self.file_modified.notify()

  def run_command_thread_fn(self):
    while True:
      with self.file_modified:
        self.file_modified.wait()
        last_modified_file = self.last_modified_file
      if last_modified_file:
        print(f"""
File {last_modified_file} modified, rerunning command...
""")
      new_argv = [arg for arg in sys.argv if arg not in ("-i", "--interactive")]
      main_command = " ".join(new_argv[1:])
      subprocess.run(f"tp {main_command}", shell=True, check=False)
      print(f"""
Done running `tp {main_command}`.
""")


def watch_directory(project_dir, command_context):
  # Load gitignore patterns
  gitignore_patterns = []
  gitignore_path = os.path.join(project_dir, ".gitignore")
  if os.path.exists(gitignore_path):
    with open(gitignore_path) as f:
      gitignore_patterns = f.readlines()

  # Create PathSpec object from gitignore
  gitignore_spec = PathSpec.from_lines(GitWildMatchPattern, gitignore_patterns)

  event_handler = FileChangeHandler(command_context, gitignore_spec)
  observer = Observer()
  observer.schedule(event_handler, project_dir, recursive=True)
  observer.start()

  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    observer.stop()
  observer.join()


def main():
  parser = argparse.ArgumentParser(
    description="tp is a CLI for common torchprime workflows."
  )
  parser.add_argument(
    "-i",
    "--interactive",
    action="store_true",
    help="Re-run the command whenever a file is edited (useful for fast dev/test iteration)",
  )

  subparsers = parser.add_subparsers(
    dest="command", required=True, help="sub-command help"
  )

  # `use` command
  parser_use = subparsers.add_parser(
    "use", help=use.__doc__, formatter_class=argparse.RawTextHelpFormatter
  )
  parser_use.add_argument("--cluster", required=True, help="Name of the XPK cluster")
  parser_use.add_argument(
    "--project", required=True, help="GCP project the cluster belongs to"
  )
  parser_use.add_argument(
    "--zone", required=True, help="Compute zone the cluster is located in"
  )
  parser_use.add_argument(
    "--num-slices",
    type=int,
    default=1,
    help="Number of TPU slice to use by default. Defaults to 1",
  )
  parser_use.add_argument(
    "--tpu-type",
    required=True,
    help="The TPU accelerator type in each slice. E.g. v6e-256 for a 256 chip Trillium pod",
  )
  parser_use.add_argument(
    "--artifact-dir",
    required=True,
    help="A Google Cloud Storage directory where artifacts such as profiles will be stored. \nE.g. gs://foo/bar",
  )
  parser_use.add_argument(
    "--upload-metrics",
    action="store_true",
    help="If given, uploads metrics to the database ",
  )
  parser_use.add_argument(
    "--bq-project", default="tpu-pytorch", help="A bigquery project to upload metrics."
  )
  parser_use.add_argument(
    "--bq-dataset",
    default="benchmark_dataset_test",
    help="A bigquery dataset to upload metrics.",
  )
  parser_use.add_argument(
    "--bq-table",
    default="benchmark_experiment",
    help="A bigquery table to upload metrics.",
  )
  parser_use.add_argument(
    "--docker-project",
    default=None,
    help="GCP project to upload docker containers to. If not set, defaults to the cluster's\n    GCP project",
  )
  parser_use.set_defaults(func=use)

  # `docker-run` command
  parser_docker_run = subparsers.add_parser("docker-run", help=docker_run.__doc__)
  parser_docker_run.add_argument(
    "--use-hf", action="store_true", help="Use HuggingFace transformer"
  )
  parser_docker_run.set_defaults(func=docker_run)

  # `run` command
  parser_run = subparsers.add_parser("run", help=run.__doc__)
  parser_run.add_argument(
    "--name",
    default=None,
    help="Name of the workload (jobset). If not specified, \ndefaults to one based on the date and time.",
  )
  parser_run.add_argument(
    "--priority",
    default="medium",
    help="Priority of the workload (jobset). If not specified, defaults to 'medium'.",
  )
  parser_run.add_argument(
    "--base-docker-url",
    default=None,
    help="If specified, `tp run` will use this PyTorch/XLA base docker image instead of \nthe one pinned inside `pyproject.toml`",
  )
  parser_run.add_argument(
    "--num-slices",
    type=int,
    default=None,
    help="Temporarily override the number of TPU slice to use for this run. \nIf unspecified, `tp run` will use the slice count configured in `tp use`.",
  )
  parser_run.add_argument(
    "--use-hf", action="store_true", help="Use HuggingFace transformer"
  )
  parser_run.add_argument(
    "--use-local-wheel",
    action="store_true",
    help="Use local torch and torch_xla wheels under folder local_dist/",
  )
  parser_run.add_argument(
    "--comments",
    default=None,
    help="Optional description of the training run, stored in the database.",
  )
  parser_run.set_defaults(func=run)

  # `test` command
  parser_test = subparsers.add_parser("test", help=test.__doc__)
  parser_test.set_defaults(func=test)

  # `doctor` command
  parser_doctor = subparsers.add_parser("doctor", help=doctor.__doc__)
  parser_doctor.set_defaults(func=doctor)

  # `save-hf-model-files-to-gcs` command
  parser_save_hf = subparsers.add_parser(
    "save-hf-model-files-to-gcs",
    help=save_hf_model_files_to_gcs.__doc__,
    formatter_class=argparse.RawTextHelpFormatter,
  )
  parser_save_hf.add_argument(
    "--repo-id",
    type=str,
    required=True,
    help="Hugging Face model or tokenizer repo ID (e.g., 'meta-llama/Llama-3-8B-hf').",
  )
  parser_save_hf.add_argument(
    "--gcs-path",
    type=str,
    required=True,
    help="Target GCS path for the model files (e.g., 'gs://bucket/models/Llama-3-8B-hf').",
  )
  parser_save_hf.add_argument(
    "--file-type",
    type=str,
    choices=["tokenizer", "model", "all"],
    default="all",
    help="Type of files to save. 'tokenizer' for tokenizer files, 'model' for model weights and configs, 'all' for both.",
  )
  parser_save_hf.add_argument(
    "--temp-dir",
    type=str,
    default=None,
    help="Path to a temporary directory with sufficient space. Defaults to system temp.",
  )
  parser_save_hf.set_defaults(func=save_hf_model_files_to_gcs)

  # Parse arguments
  known_args, remaining_args = parser.parse_known_args()

  func_to_run = known_args.func
  is_interactive = known_args.interactive

  # Prepare arguments for the function call
  func_kwargs = vars(known_args)
  del func_kwargs["command"]
  del func_kwargs["func"]
  del func_kwargs["interactive"]

  if func_to_run in [docker_run, run, test]:
    func_kwargs["args"] = remaining_args

  def command_to_execute():
    func_to_run(**func_kwargs)

  if is_interactive and func_to_run in [run, test, doctor]:
    project_dir = get_project_dir()
    print(f"Watching directory {project_dir} for changes. Press Ctrl+C to stop.\n")
    watch_directory(project_dir, None)
  else:
    command_to_execute()


if __name__ == "__main__":
  main()
