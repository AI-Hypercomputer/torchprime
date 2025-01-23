#!/usr/bin/env bash

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

cd "${SCRIPT_DIR}"

if groups "$USER" | grep -qw "docker"; then
    SUDO=""
else
    SUDO="sudo"
fi

# These are overridable by env vars
: "${TORCHPRIME_PROJECT_ID:=tpu-pytorch}"
: "${TORCHPRIME_DOCKER_URL:=}"
: "${TORCHPRIME_DOCKER_TAG:=}"

DATETIME_STR=$(date +%Y%m%d-%H%M%S)
# Generate 4 random alphabetical characters
RANDOM_CHARS="$(tr -dc 'a-z' < /dev/urandom 2>/dev/null | head -c4)"
DEFAULT_DOCKER_TAG="$DATETIME_STR-$RANDOM_CHARS"
DOCKER_TAG="${TORCHPRIME_DOCKER_TAG:-$DEFAULT_DOCKER_TAG}"
DEFAULT_DOCKER_URL="gcr.io/$TORCHPRIME_PROJECT_ID/torchprime-$USER:$DOCKER_TAG"
DOCKER_URL="${TORCHPRIME_DOCKER_URL:-$DEFAULT_DOCKER_URL}"

echo
echo "Will build a docker image and upload to: $DOCKER_URL"
echo

set -ex

$SUDO docker build --network=host --progress=auto -t "$DOCKER_TAG" ../../ -f Dockerfile
$SUDO docker tag "$DOCKER_TAG" "$DOCKER_URL"
$SUDO docker push "$DOCKER_URL"
