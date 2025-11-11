#!/usr/bin/env bash
set -euo pipefail
: "${BUCKET:?set BUCKET}"
for p in data/raw data/interim manifests; do
  gsutil -m cp -n /etc/hosts "gs://${BUCKET}/${p}/.keep" >/dev/null 2>&1 || true
done
echo "GCS prefixes ensured in gs://${BUCKET}/{data/raw,data/interim,manifests}"
