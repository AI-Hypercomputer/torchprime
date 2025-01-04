#!/usr/bin/env bash

set -eo

##### Runs SPMD training as an xpk job on a GKE cluster #####
#
# To run this, a _source_ install of xpk is required to access the latest TPU.
#
# Example: pip install git+https://github.com/AI-Hypercomputer/xpk.git@main
#

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

pushd "$SCRIPT_DIR"

# Always build a new image. This is fast when cached.
./buildpush.sh

# You can override these by setting corresponding environment variables.
: "${CLUSTER_NAME:=bodaborg-v6e-256}"
: "${DOCKER_URL:=gcr.io/tpu-pytorch/llama3:latest}"
: "${NUM_SLICES:=2}"
: "${TPU_TYPE:=v6e-256}"
: "${ZONE:=us-east5-c}"
: "${PROJECT_ID:=tpu-prod-env-automated}"

DATETIMESTR=$(date +%Y%m%d-%H%M%S)
COMMAND="python launcher/run_xpk.py $@"

# Forward `HF_TOKEN` (HuggingFace token) env var if set.
HF_TOKEN_ARG=()
if [[ -n "$HF_TOKEN" ]]; then
    HF_TOKEN_ARG=("--env" "HF_TOKEN=$HF_TOKEN")
fi

xpk workload create \
    --cluster ${CLUSTER_NAME} \
    --docker-image ${DOCKER_URL} \
    --workload "${USER}-xpk-${TPU_TYPE}-${NUM_SLICES}-${DATETIMESTR}" \
    --tpu-type=${TPU_TYPE} \
    --num-slices=${NUM_SLICES} \
    --zone $ZONE \
    --project $PROJECT_ID \
    --enable-debug-logs \
    "${HF_TOKEN_ARG[@]}" \
    --command "$COMMAND"
