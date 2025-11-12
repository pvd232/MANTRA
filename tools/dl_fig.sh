#!/usr/bin/env bash
set -euo pipefail

# Usage: dl_one.sh <figshare_download_url> <dest_relpath> [--bucket BUCKET] [--prefix PREFIX]
# Example:
#   dl_one.sh "https://plus.figshare.com/ndownloader/files/35775606" "K562_gwps/rpe1_raw_singlecell_01.h5ad" \
#              --bucket mantra-mlfg-prod-uscentral1-8e7a --prefix data/raw

URL="${1:?need download URL}"
OUT="${2:?need destination relative path (e.g. K562_gwps/file.h5ad)}"
shift 2 || true

BUCKET=                           # e.g. mantra-mlfg-prod-uscentral1-8e7a  (no gs://)
PREFIX="data/raw"                 # gs subdir prefix
WORKDIR="${WORKDIR:-/tmp/mantra_downloads}"
RETRIES="${RETRIES:-20}"

# parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket) BUCKET="$2"; shift 2;;
    --prefix) PREFIX="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# choose downloader: single-stream (presigned S3 links expire fast)
if command -v aria2c >/dev/null 2>&1; then
  DL="aria2c --check-integrity=false --continue=true -x1 -s1 --retry-wait=3 --max-tries=${RETRIES} -o"
else
  DL="curl -L --fail --retry ${RETRIES} --retry-all-errors --retry-delay 3 -C - -o"
fi

fname="$(basename "$OUT")"
dest_local="${WORKDIR}/${fname}"

echo "==> Fetch: $URL"
# shellcheck disable=SC2086
if ! eval ${DL} "\"${fname}\"" "\"${URL}\""; then
  echo "   primary failed; retrying with curl single-stream…"
  curl -L --fail --retry "${RETRIES}" --retry-all-errors --retry-delay 3 -C - -o "${dest_local}" "${URL}"
fi

[[ -s "$dest_local" ]] || { echo "!! empty download: $dest_local" >&2; exit 1; }

if [[ -n "$BUCKET" ]]; then
  command -v gsutil >/dev/null 2>&1 || { echo "missing gsutil" >&2; exit 1; }
  gs_uri="gs://${BUCKET%/}/${PREFIX%/}/${OUT}"
  # remove any 0-byte object so -n doesn't “skip” a bad prior upload
  if gsutil stat "$gs_uri" >/dev/null 2>&1; then
    size=$(gsutil stat "$gs_uri" | awk '/Content-Length:/ {print $3}')
    [[ "${size:-}" == "0" ]] && gsutil rm -f "$gs_uri" || true
  fi
  echo "==> Upload: $dest_local -> $gs_uri"
  gsutil -m -o "GSUtil:parallel_composite_upload_threshold=150M" cp -n "$dest_local" "$gs_uri"
fi

echo "[ok] $dest_local"
