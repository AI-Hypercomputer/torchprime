#!/usr/bin/env bash

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

pushd "${SCRIPT_DIR}"

if groups "$USER" | grep -qw "docker"; then
    SUDO=""
else
    SUDO="sudo"
fi

$SUDO docker build --network=host -t llama3 ../ -f Dockerfile
$SUDO docker tag llama3 gcr.io/tpu-pytorch/llama3:latest
$SUDO docker push gcr.io/tpu-pytorch/llama3:latest
