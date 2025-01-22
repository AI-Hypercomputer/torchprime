"""
tp is a CLI for common torchprime workflows.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
import toml
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Config:
  cluster: str
  project: str
  zone: str
  num_slices: int
  tpu_type: str
  artifact_dir: str


@click.group()
def cli():
  """
  tp is a CLI for common torchprime workflows.
  """
  pass


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
def use(
  cluster: str,
  project: str,
  zone: str,
  num_slices: int,
  tpu_type: str,
  artifact_dir: str,
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
  )
  gcloud_config_name = f"torchprime-{project}-{zone}"
  create_and_activate_gcloud(gcloud_config_name, config)
  assert artifact_dir.startswith(
    "gs://"
  ), f"{artifact_dir} must be in a GCS bucket (start with gs://)"

  path = write_config(config)
  click.echo(f"Written config {path.relative_to(os.getcwd())}")


def create_and_activate_gcloud(gcloud_config_name, config: Config):
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
  if existing:
    subprocess.check_output(
      [
        "gcloud",
        "config",
        "configurations",
        "activate",
        gcloud_config_name,
      ]
    )
  else:
    subprocess.check_output(
      ["gcloud", "config", "configurations", "create", gcloud_config_name, "--activate"]
    )

  subprocess.check_output(
    [
      "gcloud",
      "config",
      "set",
      "compute/zone",
      config.zone,
    ]
  )
  subprocess.check_output(
    [
      "gcloud",
      "config",
      "set",
      "project",
      config.project,
    ]
  )


@cli.command(
  context_settings=dict(
    ignore_unknown_options=True,
  )
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def run(args):
  """
  Runs the provided SPMD training command as an xpk job on a GKE cluster.
  """
  config = read_config()

  click.echo(get_project_dir().absolute())

  # Build docker image.
  assert os.system(Path(__file__).parent / "buildpush.sh") == 0

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
      "gcr.io/tpu-pytorch/llama3:latest",
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


if __name__ == "__main__":
  cli()
