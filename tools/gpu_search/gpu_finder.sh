#!/usr/bin/env bash
set -euo pipefail

PROJECT=mantra-477901
REGION=us-west1
IMAGE_FAM=pytorch-2-7-cu128-ubuntu-2204-nvidia-570

# H100: a3-highgpu-1g (1x H100 80GB)
MACHINE_TYPE=a3-highgpu-1g
DISK_SIZE=500GB

for Z in ${REGION}-a ${REGION}-b ${REGION}-c; do
  echo ">>> Trying $Z"
  gcloud compute instances create mantra-h100 \
    --project="${PROJECT}" \
    --zone="${Z}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator=count=1,type=nvidia-h100-80gb \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --maintenance-policy=TERMINATE \
    --image-project=deeplearning-platform-release \
    --image-family="${IMAGE_FAM}" \
    --boot-disk-type=pd-ssd \
    --boot-disk-size="${DISK_SIZE}" \
    --scopes=cloud-platform \
    --network-interface=network=default,subnet=default,no-address \
  && { echo "✅ Landed in $Z"; break; } \
  || echo "✗ $Z failed (check error above)"
done
