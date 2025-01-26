"""
tp is a CLI for common torchprime workflows.
"""

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
import toml
from dataclasses_json import dataclass_json
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern  # type: ignore
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import torchprime.launcher.doctor
from torchprime.launcher.buildpush import buildpush


@dataclass_json
@dataclass
class Config:
  cluster: str
  project: str
  zone: str
  num_slices: int
  tpu_type: str
  artifact_dir: str
  docker_project: str | None = None


def interactive(f):
  @click.pass_context
  def wrapper(ctx, *args, **kwargs):
    return run_with_watcher(ctx)(f)(*args, **kwargs)

  wrapper.__name__ = f.__name__
  wrapper.__doc__ = f.__doc__
  return wrapper


@click.group()
@click.option(
  "-i",
  "--interactive",
  is_flag=True,
  default=False,
  help="Re-run the command whenever a file is edited (useful for fast dev/test iteration)",
)
@click.pass_context
def cli(ctx, interactive):
  """
  tp is a CLI for common torchprime workflows.
  """
  ctx.ensure_object(dict)
  ctx.obj["interactive"] = interactive


@cli.command()
@click.option("--cluster", required=True, help="Name of the XPK cluster")
@click.option("--project", required=True, help="GCP project the cluster belongs to")
@click.option("--zone", required=True, help="Compute zone the cluster is located in")
@click.option(
  "--num-slices",
  required=False,
  default=1,
  help="Number of TPU slice to use. Defaults to 1",
)
@click.option(
  "--tpu-type",
  required=True,
  help="The TPU accelerator type in each slice. E.g. v6e-256 for a 256 chip Trillium pod",
)
@click.option(
  "--artifact-dir",
  required=True,
  help="A Google Cloud Storage directory where artifacts such as profiles will be stored. \
E.g. gs://foo/bar",
)
@click.option(
  "--docker-project",
  required=False,
  default=None,
  help="GCP project to upload docker containers to. If not set, defaults to the cluster's\
    GCP project",
)
def use(
  cluster: str,
  project: str,
  zone: str,
  num_slices: int,
  tpu_type: str,
  artifact_dir: str,
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
    docker_project=docker_project,
  )
  gcloud_config_name = f"torchprime-{project}-{zone}"
  create_and_activate_gcloud(gcloud_config_name, config)
  assert artifact_dir.startswith(
    "gs://"
  ), f"{artifact_dir} must be in a GCS bucket (start with gs://)"

  path = write_config(config)
  click.echo(f"Written config {path.relative_to(os.getcwd())}")
  torchprime.launcher.doctor.check_all()


def create_and_activate_gcloud(gcloud_config_name, config: Config):
  click.echo("Activating gcloud config...")
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
      "auth",
      "application-default",
      "set-quota-project",
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


@cli.command(
  context_settings=dict(
    ignore_unknown_options=True,
  )
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@interactive
def run(args):
  """
  Runs the provided SPMD training command as an xpk job on a GKE cluster.
  """
  config = read_config()

  click.echo(get_project_dir().absolute())

  # Build docker image.
  docker_project = config.docker_project
  if docker_project is None:
    docker_project = config.project
  docker_url = buildpush(docker_project)

  # Submit xpk workload
  datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
  command = ["python", "torchprime/launcher/thunk.py"] + list(args)

  # Forward a bunch of important env vars.
  env_forwarding = [
    *forward_env("HF_TOKEN"),  # HuggingFace token
    *forward_env("XLA_IR_DEBUG"),  # torch_xla debugging flag
    *forward_env("XLA_HLO_DEBUG"),  # torch_xla debugging flag
  ]

  # Pass artifact dir as another env var.
  artifact_arg = ["--env", f"TORCHPRIME_ARTIFACT_DIR={config.artifact_dir}"]

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
      f"{os.environ['USER']}-xpk-{config.tpu_type}-{config.num_slices}-{datetime_str}",
      "--tpu-type",
      config.tpu_type,
      "--num-slices",
      str(config.num_slices),
      "--zone",
      config.zone,
      "--project",
      config.project,
      "--enable-debug-logs",
    ]
    + env_forwarding
    + artifact_arg
    + ["--command", " ".join(command)]
  )
  subprocess.run(xpk_command, check=True)


@cli.command(
  context_settings=dict(
    ignore_unknown_options=True,
  )
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@interactive
def test(args):
  """
  Runs unit tests in torchprime by forwarding arguments to pytest.
  """
  ensure_command("pytest")
  try:
    subprocess.run(["pytest"] + list(args), check=True)
  except subprocess.CalledProcessError as e:
    sys.exit(e.returncode)


@cli.command()
@interactive
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
      click.echo("Previous command outputs:")
      click.echo(self.outputs.decode("utf-8"))
      click.echo()
      click.echo(f"❌ Error running `{' '.join(command)}` ❌")
      click.echo()
      click.echo(e.stdout)
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
        click.echo(f"""
File {last_modified_file} modified, rerunning command...
""")
      sys.argv[1] = sys.argv[1].replace("-i", "").replace("--interactive", "").strip()
      main_command = " ".join(s for s in sys.argv[1:] if s != "")
      subprocess.run(f"tp {main_command}", shell=True, check=False)
      click.echo(f"""
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


def run_with_watcher(ctx):
  """Wrapper to run commands with file watching if interactive mode is enabled"""

  def decorator(f):
    def wrapper(*args, **kwargs):
      # If interactive mode is enabled, start watching for changes
      if ctx.obj.get("interactive"):
        project_dir = get_project_dir()
        click.echo(
          f"Watching directory {project_dir} for changes. Press Ctrl+C to stop.\n"
        )
        watch_directory(project_dir, ctx)
      else:
        # Just run the command
        return f(*args, **kwargs)

    return wrapper

  return decorator


if __name__ == "__main__":
  cli()
