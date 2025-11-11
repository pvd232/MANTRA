#!/usr/bin/env bash
set -euo pipefail

# ---- config you can edit or export beforehand ----
PROJECT="${PROJECT:-mantra-477901}"
ZONE="${ZONE:-us-west4-a}"
VM="${VM:-mantra-g2}"
BUCKET="${BUCKET:-mantra-mlfg-prod-uscentral1-8e7a}"

export PROJECT ZONE VM BUCKET

echo "[*] Bootstrapping VM…"
make bootstrap PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM" BUCKET="$BUCKET"

echo "[*] Ensure GCS prefixes exist…"
make gcs.init PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM" BUCKET="$BUCKET"

echo "[*] Downloading figshare files -> GCS…"
make data.download PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM" BUCKET="$BUCKET"

echo "[*] CUDA/Torch sanity on VM…"
make vm.cuda PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM"

echo "[*] Run MANTRA pipeline on VM…"
make vm.run PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM" PIPELINE=all

echo "[✓] Done."