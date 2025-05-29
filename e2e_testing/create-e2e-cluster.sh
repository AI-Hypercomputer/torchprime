#!/usr/bin/env bash

# Specify zones, number of slices, and the cluster name.
CLUSTER_NAME=tpu-v6e-ci
ZONE=southamerica-west1-a
REGION=southamerica-west1
PROJECT=tpu-prod-env-one-vm
TPU_TYPE=v6e-4
NUM_SLICES=128

NETWORK_NAME_1=${CLUSTER_NAME}-mtu9k-1-${ZONE}
SUBNET_NAME_1=${CLUSTER_NAME}-privatesubnet-1-${ZONE}
FIREWALL_RULE_NAME=${CLUSTER_NAME}-privatefirewall-1-${ZONE}
ROUTER_NAME=${CLUSTER_NAME}-network-1-${ZONE}
NAT_CONFIG=${CLUSTER_NAME}-natconfig-1-${ZONE}

# Use a custom network for better performance as well as avoid the default network to be overloaded.
gcloud compute networks create "${NETWORK_NAME_1}" --mtu=8896 --bgp-routing-mode=regional --subnet-mode=custom --project=$PROJECT
gcloud compute networks subnets create "${SUBNET_NAME_1}" --network="${NETWORK_NAME_1}" --range=10.11.0.0/18 --region="${REGION}" --project=$PROJECT
gcloud compute firewall-rules create "${FIREWALL_RULE_NAME}" --network "${NETWORK_NAME_1}" --allow tcp,icmp,udp --project="${PROJECT}"
gcloud compute routers create "${ROUTER_NAME}" \
  --project="${PROJECT}" \
  --network="${NETWORK_NAME_1}" \
  --region="${REGION}"
gcloud compute routers nats create "${NAT_CONFIG}" \
  --router="${ROUTER_NAME}" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges \
  --project="${PROJECT}" \
  --enable-logging

export CLUSTER_ARGUMENTS="--enable-dataplane-v2 --enable-ip-alias --enable-multi-networking --network=${NETWORK_NAME_1} --subnetwork=${SUBNET_NAME_1}"

xpk cluster create \
    --tpu-type ${TPU_TYPE} \
    --cluster ${CLUSTER_NAME} \
    --num-slices ${NUM_SLICES} \
    --on-demand \
    --zone ${ZONE} \
    --project ${PROJECT} \
    --custom-cluster-arguments="${CLUSTER_ARGUMENTS}" \
    --default-pool-cpu-machine-type=n2-standard-32
