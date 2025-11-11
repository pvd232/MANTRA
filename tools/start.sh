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

echo "[*] Ensure local workspace dirs…"
make vm.ensure_dirs PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM"

echo "[*] Downloading figshare files -> GCS…"
make data.download PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM" BUCKET="$BUCKET"

echo "[*] CUDA/Torch sanity on VM…"
make vm.cuda PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM"

echo "[*] Run MANTRA pipeline on VM…"
make vm.run PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM" PIPELINE=all

echo "[*] Clean staging downloads on VM…"
make cleanup.staging PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM"

echo "[*] Pull RAW and SMR from GCS to VM (immutable)"
make sync.all.down PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM"


echo "[*] Push INTERIM + OUT back to GCS"
make sync.all.up PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM"
