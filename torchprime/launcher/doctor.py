"""
Doctor checks for essential programs needed to launch distributed training.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import click


class CheckFailedError(Exception):
  pass


def check_gcr_io():
  """Check that docker config contains gcr.io credential helper."""
  try:
    docker_config = json.loads(
      Path(os.path.expanduser("~/.docker/config.json")).read_text()
    )
  except FileNotFoundError as e:
    raise CheckFailedError("docker config not found. Please install docker.") from e
  try:
    cred_helpers = docker_config["credHelpers"]
    _gcr_io = cred_helpers["gcr.io"]
  except KeyError:
    raise CheckFailedError(
      """
Did not find a handler for `gcr.io` in docker credential helpers.

TorchPrime uploads docker containers to the `gcr.io` docker registry, which
requires valid credentials.

To setup the credentials, please run:

  gcloud auth configure-docker

""".lstrip()
    ) from None
  try:
    subprocess.run(
      ["gcloud", "auth", "configure-docker", "gcr.io"], check=True, capture_output=True
    )
  except subprocess.CalledProcessError as e:
    raise CheckFailedError(
      f"gcloud auth configure-docker gcr.io failed: {e.stderr.decode()}"
    ) from e


def check_gcloud_auth_login():
  """Check that gcloud is logged in."""
  try:
    subprocess.run(
      ["gcloud", "auth", "print-access-token"], check=True, capture_output=True
    )
  except subprocess.CalledProcessError as e:
    raise CheckFailedError(
      f"gcloud auth print-access-token failed: {e.stderr.decode()}"
    ) from e


def check_kubectl():
  """Check that kubectl is installed."""
  try:
    subprocess.run(["kubectl", "help"], check=True, capture_output=True)
  except FileNotFoundError:
    raise CheckFailedError(
      f"""kubectl not found.

{get_kubectl_install_instructions()}
""".strip()
    ) from None


def check_gke_gcloud_auth_plugin():
  """Check that gke-gcloud-auth-plugin is installed."""
  try:
    subprocess.run(
      ["gcloud", "components", "list", "--filter=gke-gcloud-auth-plugin"],
      check=True,
      capture_output=True,
    )
  except subprocess.CalledProcessError as e:
    raise CheckFailedError(
      f"gcloud components list --filter=gke-gcloud-auth-plugin failed: {e.stderr.decode()}"
    ) from e


def check_all():
  click.echo("Checking environment...")
  for check in [
    check_gcloud_auth_login,
    check_gcr_io,
    check_kubectl,
    check_gke_gcloud_auth_plugin,
  ]:
    assert check.__doc__ is not None
    click.echo(check.__doc__ + "..", nl=False)
    try:
      check()
    except CheckFailedError as e:
      click.echo()
      click.echo()
      click.echo(f"❌ Error during {check.__name__} ❌")
      click.echo(e)
      sys.exit(-1)
    click.echo(" ✅")
  click.echo(
    "🎉 All checks passed. You should be ready to launch distributed training."
  )


def get_kubectl_install_instructions():
  # If gcloud is installed via `apt`, then we should do the same for `kubectl`.
  if is_package_installed("google-cloud-cli"):
    return """
Since `gcloud` is installed with `apt`, please install `kubectl` with:

  sudo apt install kubectl

""".lstrip()

  # Otherwise, point users to the GKE docs.
  return "Please visit \
https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_kubectl"


def is_package_installed(package_name):
  try:
    # Run the dpkg-query command to check for the package
    subprocess.run(
      ["dpkg-query", "-W", "-f='${Status}'", package_name],
      check=True,
      capture_output=True,
    )
    return True
  except subprocess.CalledProcessError:
    return False


if __name__ == "__main__":
  check_all()
