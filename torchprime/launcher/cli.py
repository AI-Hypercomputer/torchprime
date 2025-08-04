"""
tp is a CLI for common torchprime workflows.
"""

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
from absl import app, flags
from dataclasses_json import dataclass_json
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from rich.text import Text
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import torchprime.launcher.doctor
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


FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    "interactive",
    False,
    "Re-run the command whenever a file is edited (useful for fast dev/test iteration)",
)

# Flags for `use` command
flags.DEFINE_string("cluster", None, "Name of the XPK cluster")
flags.DEFINE_string("project", None, "GCP project the cluster belongs to")
flags.DEFINE_string("zone", None, "Compute zone the cluster is located in")
flags.DEFINE_integer("num-slices", 1, "Number of TPU slice to use by default. Defaults to 1")
flags.DEFINE_string("tpu-type", None, "The TPU accelerator type in each slice. E.g. v6e-256 for a 256 chip Trillium pod")
flags.DEFINE_string("artifact-dir", None, "A Google Cloud Storage directory where artifacts such as profiles will be stored. E.g. gs://foo/bar")
flags.DEFINE_boolean("upload-metrics", False, "If given, uploads metrics to the database ")
flags.DEFINE_string("bq-project", "tpu-pytorch", "A bigquery project to upload metrics.")
flags.DEFINE_string("bq-dataset", "benchmark_dataset_test", "A bigquery dataset to upload metrics.")
flags.DEFINE_string("bq-table", "benchmark_experiment", "A bigquery table to upload metrics.")
flags.DEFINE_string("docker-project", None, "GCP project to upload docker containers to. If not set, defaults to the cluster's GCP project")

# Flags for `run` command
flags.DEFINE_string("name", None, "Name of the workload (jobset). If not specified, defaults to one based on the date and time.")
flags.DEFINE_string("base-docker-url", None, "If specified, `tp run` will use this PyTorch/XLA base docker image instead of the one pinned inside `pyproject.toml`")
flags.DEFINE_boolean("use-hf", False, "Use HuggingFace transformer")
flags.DEFINE_boolean("use-local-wheel", False, "Use local torch and torch_xla wheels under folder local_dist/")
flags.DEFINE_string("comments", None, "Optional description of the training run, stored in the database.")


def use():
  """
  Sets up various config like XPK cluster name, GCP project, etc for all
  subsequent commands to use. Typically, you would only run this command once when
  you first clone the repo, or when switching to a different hardware/cluster.

  This will also create and activate a gcloud configuration so that you don't
  have to type the project and zone if you drop down to xpk.
  """
  config = Config(
    cluster=FLAGS.cluster,
    project=FLAGS.project,
    zone=FLAGS.zone,
    num_slices=FLAGS['num-slices'].value,
    tpu_type=FLAGS['tpu-type'].value,
    artifact_dir=FLAGS['artifact-dir'].value,
    upload_metrics=FLAGS['upload-metrics'].value,
    bq_project=FLAGS['bq-project'].value,
    bq_dataset=FLAGS['bq-dataset'].value,
    bq_table=FLAGS['bq-table'].value,
    docker_project=FLAGS['docker-project'].value,
  )
  gcloud_config_name = f"torchprime-{FLAGS.project}-{FLAGS.zone}"
  create_and_activate_gcloud(gcloud_config_name, config)
  assert FLAGS['artifact-dir'].value.startswith("gs://"), (
    f"{FLAGS['artifact-dir'].value} must be in a GCS bucket (start with gs://)"
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


def docker_run(argv):
  """
  Runs the provided training command locally for quick testing.
  """
  print(get_project_dir().absolute())

  # Build docker image.
  build_arg = ["USE_TRANSFORMERS=true"] if FLAGS['use-hf'].value else None
  placeholder_url = "torchprime-dev:local"
  docker_url = buildpush(
    push_docker=False, placeholder_url=placeholder_url, build_arg=build_arg
  )
  # Forward a bunch of important env vars.
  env_forwarding = [
    arg for env_var in _DOCKER_ENV_FORWARD_LIST for arg in forward_env(env_var)
  ]
  args = list(v for v in argv[1:] if v != "")
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


def run(argv):
  """
  Runs the provided SPMD training command as an xpk job on a GKE cluster.
  """
  config = read_config()

  print(get_project_dir().absolute())

  # Build docker image.
  build_arg = []
  if FLAGS['use-hf'].value:
    build_arg.append("USE_TRANSFORMERS=true")
  if FLAGS['use-local-wheel'].value:
    build_arg.append("USE_LOCAL_WHEEL=true")
  docker_project = config.docker_project
  if docker_project is None:
    docker_project = config.project
  docker_url = buildpush(
    torchprime_project_id=docker_project,
    build_arg=build_arg,
    base_docker_url=FLAGS['base-docker-url'].value,
  )

  # Submit xpk workload
  workload_name = FLAGS.name
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

  command = ["python", "torchprime/launcher/thunk.py"] + list(argv[1:])

  num_slices = FLAGS['num-slices'].value
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
    f"TORCHPRIME_COMMENTS={FLAGS.comments}",
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
  styled_artifacts = Text(
    f"{config.artifact_dir}/{workload_name}", style="bold green"
  )
  print(f"""
Workload {styled_workload} submitted to cluster {styled_cluster}

Artifacts are stored at {styled_artifacts}
""")


def test(argv):
  """
  Runs unit tests in torchprime by forwarding arguments to pytest.
  """
  ensure_command("pytest")
  try:
    subprocess.run(["pytest"] + list(argv[1:]), check=True)
  except subprocess.CalledProcessError as e:
    sys.exit(e.returncode)


def doctor():
  """
  Checks for any problems in your environment (missing packages, credentials, etc.).
  """
  torchprime.launcher.doctor.check_all()


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
  default_config.write_text(toml.dumps(config.to_dict()))
  return default_config


def read_config() -> Config:
  config_path = get_config_dir() / DEFAULT_CONFIG_NAME
  if not config_path.exists():
    raise RuntimeError(f"No config found at {config_path}. Run `tp use` first.")
  return Config.from_dict(toml.load(config_path))


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
  def __init__(self, gitignore_spec):
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
    if current_time - self.last_trigger_time > 1:
      self.last_trigger_time = current_time
    else:
      return

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
      main_command = " ".join(s for s in sys.argv if s != "-i" and s != "--interactive")
      subprocess.run(f"tp {main_command}", shell=True, check=False)
      print(f"""
Done running `tp {main_command}`.
""")


def watch_directory(project_dir):
  # Load gitignore patterns
  gitignore_patterns = []
  gitignore_path = os.path.join(project_dir, ".gitignore")
  if os.path.exists(gitignore_path):
    with open(gitignore_path) as f:
      gitignore_patterns = f.readlines()

  # Create PathSpec object from gitignore
  gitignore_spec = PathSpec.from_lines(GitWildMatchPattern, gitignore_patterns)

  event_handler = FileChangeHandler(gitignore_spec)
  observer = Observer()
  observer.schedule(event_handler, project_dir, recursive=True)
  observer.start()

  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    observer.stop()
  observer.join()


def main(argv):
  if len(argv) < 2:
    print("Usage: tp <command> [options]")
    return

  command = argv[1]
  if FLAGS.interactive:
    project_dir = get_project_dir()
    print(
      f"Watching directory {project_dir} for changes. Press Ctrl+C to stop.\n"
    )
    watch_directory(project_dir)
    return

  if command == "use":
    use()
  elif command == "docker-run":
    docker_run(argv)
  elif command == "run":
    run(argv)
  elif command == "test":
    test(argv)
  elif command == "doctor":
    doctor()
  else:
    print(f"Unknown command: {command}")
    return


def cli():
  app.run(main)

if __name__ == "__main__":
  cli()
