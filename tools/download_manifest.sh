#!/usr/bin/env bash
set -euo pipefail

# Usage: download_from_manifest.sh /path/to/manifest.csv <BUCKET_NAME> [DEST_PREFIX]
# Example: download_from_manifest.sh ~/MANTRA/configs/download_manifest.csv mantra-mlfg-prod-uscentral1-8e7a data/raw
MANIFEST=${1:?manifest csv required}
BUCKET=${2:?bucket required}
DEST_PREFIX=${3:-data/raw}

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# Normalize CRLF just in case
sed -i 's/\r$//' "$MANIFEST"

echo "[info] Using bucket: gs://${BUCKET}/${DEST_PREFIX}"
echo "[info] Manifest: $MANIFEST"

# Read CSV: url,dst_relpath,sha256(optional)
# Skip header if present
while IFS=, read -r url dst_relpath sha256sum_opt; do
  # skip blank lines / header
  [[ -z "${url// }" ]] && continue
  [[ "$url" == "url" ]] && continue

  # strip leading slash from dst_relpath
  dst_relpath="${dst_relpath#/}"

  fname="$(basename "$url")"
  out="${TMPDIR}/${fname}"

  echo "[info] Downloading: $url"
  curl -fL --retry 3 --retry-delay 2 -o "$out" "$url"

  if [[ -n "${sha256sum_opt// }" ]]; then
    echo "[info] Verifying sha256 for ${fname}"
    # expects the third column is the raw sha256 hex
    echo "${sha256sum_opt}  ${out}" | sha256sum -c -
  else
    echo "[warn] No sha256 provided for ${fname} (skipping verification)"
  fi

  # Destination is full object path including filename as provided in dst_relpath
  gs_uri="gs://${BUCKET}/${DEST_PREFIX}/${dst_relpath}"
  echo "[info] Uploading -> ${gs_uri}"
  # -n = no clobber (keeps existing objects)
  gsutil -m cp -n "$out" "$gs_uri"
done < "$MANIFEST"

echo "[done] All files processed."
