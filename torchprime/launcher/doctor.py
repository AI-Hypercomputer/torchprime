"""
Doctor checks for essential programs needed to launch distributed training.
"""

import json
import os
import subprocess
from pathlib import Path

# Check `gcloud auth configure-docker gcr.io`
# Check `gcloud auth login`
# Check `kubectl`
# Check `gke-gcloud-auth-plugin`


def check_gcr_io():
  """Check that docker config contains gcr.io credential helper."""
  try:
    docker_config = json.loads(
      Path(os.path.expanduser("~/.docker/config.json")).read_text()
    )
  except FileNotFoundError as e:
    raise RuntimeError("docker config not found. Please install docker.") from e
  try:
    cred_helpers = docker_config["credHelpers"]
    _gcr_io = cred_helpers["gcr.io"]
  except KeyError:
    pass
  try:
    subprocess.run(
      ["gcloud", "auth", "configure-docker", "gcr.io"], check=True, capture_output=True
    )
  except subprocess.CalledProcessError as e:
    raise RuntimeError(
      f"gcloud auth configure-docker gcr.io failed: {e.stderr.decode()}"
    ) from e


def check_gcloud_auth_login():
  """Check that gcloud is logged in."""
  try:
    subprocess.run(["gcloud", "auth", "list"], check=True, capture_output=True)
  except subprocess.CalledProcessError as e:
    raise RuntimeError(f"gcloud auth list failed: {e.stderr.decode()}") from e


def check_kubectl():
  """Check that kubectl is installed."""
  try:
    subprocess.run(["kubectl", "version"], check=True, capture_output=True)
  except FileNotFoundError:
    raise RuntimeError("kubectl not found. Please install it.")
  except subprocess.CalledProcessError as e:
    raise RuntimeError(f"kubectl version failed: {e.stderr.decode()}") from e


def check_gke_gcloud_auth_plugin():
  """Check that gke-gcloud-auth-plugin is installed."""
  try:
    subprocess.run(
      ["gcloud", "components", "list", "--filter=gke-gcloud-auth-plugin"],
      check=True,
      capture_output=True,
    )
  except subprocess.CalledProcessError as e:
    raise RuntimeError(
      f"gcloud components list --filter=gke-gcloud-auth-plugin failed: {e.stderr.decode()}"
    ) from e


def check_all():
  check_gcr_io()
  check_gcloud_auth_login()
  check_kubectl()
  check_gke_gcloud_auth_plugin()


if __name__ == "__main__":
  check_all()
