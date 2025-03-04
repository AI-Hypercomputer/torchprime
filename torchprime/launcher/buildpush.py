#!/usr/bin/env python3

import datetime
import getpass
import grp
import os
import random
import string
import subprocess
from pathlib import Path

import click


def buildpush(
  torchprime_project_id,
  torchprime_docker_url=None,
  torchprime_docker_tag=None,
  *,
  build_arg=None,
) -> str:
  # Determine the path of this script and its directory
  script_path = os.path.realpath(__file__)
  script_dir = Path(os.path.dirname(script_path))
  context_dir = script_dir.parent.parent.relative_to(os.getcwd())
  docker_file = (script_dir / "Dockerfile").relative_to(os.getcwd())

  # Check if the user is in the 'docker' group
  user = getpass.getuser()
  groups_for_user = [g.gr_name for g in grp.getgrall() if user in g.gr_mem]
  sudo_cmd = "" if "docker" in groups_for_user else "sudo"

  # Generate date/time string and 4 random lowercase letters
  datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  random_chars = "".join(random.choices(string.ascii_lowercase, k=4))

  # Determine Docker tag
  default_tag = f"{datetime_str}-{random_chars}"
  docker_tag = torchprime_docker_tag if torchprime_docker_tag else default_tag

  # Determine Docker URL
  default_url = f"gcr.io/{torchprime_project_id}/torchprime-{user}:{docker_tag}"
  docker_url = torchprime_docker_url if torchprime_docker_url else default_url

  print()
  print(f"Will build a docker image and upload to: {docker_url}")
  print()

  build_cmd = f"{sudo_cmd} docker build"
  if build_arg:
    for _arg in build_arg:
      build_cmd += f" --build-arg {_arg}"
  build_cmd += (
    f" --network=host --progress=auto -t {docker_tag} {context_dir} -f {docker_file}"
  )

  # Build, tag, and push Docker image
  try:
    _run(
      build_cmd,
    )
    _run(
      f"{sudo_cmd} docker tag {docker_tag} {docker_url}",
    )
    _run(f"{sudo_cmd} docker push {docker_url}")
  except subprocess.CalledProcessError as e:
    print(f"Error running command: {e}")
    exit(e.returncode)

  return docker_url


def _run(command):
  click.echo(command)
  subprocess.run(
    command,
    shell=True,
    check=True,
  )


if __name__ == "__main__":
  # Read environment variables or use defaults
  torchprime_project_id = os.getenv("TORCHPRIME_PROJECT_ID", "tpu-pytorch")
  torchprime_docker_url = os.getenv("TORCHPRIME_DOCKER_URL", None)
  torchprime_docker_tag = os.getenv("TORCHPRIME_DOCKER_TAG", None)
  buildpush(
    torchprime_project_id,
    torchprime_docker_url,
    torchprime_docker_tag,
  )
