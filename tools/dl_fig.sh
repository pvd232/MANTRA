#!/usr/bin/env bash
set -euo pipefail

# Usage: dl_one.sh <figshare_download_url> <dest_rel_or_abs_path> [--bucket BUCKET] [--prefix PREFIX]
# Example:
#   dl_one.sh "https://plus.figshare.com/ndownloader/files/35775606" "data/raw/K562_gwps/rpe1_raw_singlecell_01.h5ad" \
#              --bucket mantra-mlfg-prod-uscentral1-8e7a --prefix data/raw

URL="${1:?need download URL}"
DEST_PATH="${2:?need destination path (e.g. data/raw/K562_gwps/file.h5ad)}"
shift 2 || true

BUCKET=""                         # e.g. mantra-mlfg-prod-uscentral1-8e7a  (no gs://)
PREFIX="data/raw"                 # gs subdir prefix for upload
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

# single-stream downloader (presigned S3 links expire quickly)
if command -v aria2c >/dev/null 2>&1; then
  DL="aria2c --check-integrity=false --continue=true -x1 -s1 --retry-wait=3 --max-tries=${RETRIES} -o"
else
  DL="curl -L --fail --retry ${RETRIES} --retry-all-errors --retry-delay 3 -C - -o"
fi

fname="$(basename "$DEST_PATH")"
tmp_local="${WORKDIR}/${fname}"

echo "==> Fetch: $URL"
# shellcheck disable=SC2086
if ! eval ${DL} "\"${fname}\"" "\"${URL}\""; then
  echo "   primary failed; retrying with curl single-stream…"
  curl -L --fail --retry "${RETRIES}" --retry-all-errors --retry-delay 3 -C - -o "${tmp_local}" "${URL}"
fi

[[ -s "$tmp_local" ]] || { echo "!! empty download: $tmp_local" >&2; exit 1; }

# move into final local destination (create its directory)
dest_dir="$(dirname "$DEST_PATH")"
mkdir -p "$dest_dir"
mv -f "$tmp_local" "$DEST_PATH"
echo "==> Saved locally: $DEST_PATH"

# optional upload to GCS
if [[ -n "$BUCKET" ]]; then
  command -v gsutil >/dev/null 2>&1 || { echo "missing gsutil" >&2; exit 1; }
  # if DEST_PATH is under PREFIX already, preserve relative under PREFIX; else just append filename
  rel_under_prefix="${DEST_PATH#*/}"   # naive fallback
  # prefer to preserve user-provided path relative to PREFIX:
  if [[ "$DEST_PATH" == *"/${PREFIX%/}/"* ]]; then
    rel_under_prefix="${DEST_PATH##*/${PREFIX%/}/}"
  else
    rel_under_prefix="$fname"
  fi
  gs_uri="gs://${BUCKET%/}/${PREFIX%/}/${rel_under_prefix}"

  # clean any prior 0-byte object
  if gsutil stat "$gs_uri" >/dev/null 2>&1; then
    size=$(gsutil stat "$gs_uri" | awk '/Content-Length:/ {print $3}')
    [[ "${size:-}" == "0" ]] && gsutil rm -f "$gs_uri" || true
  fi

  echo "==> Upload: $DEST_PATH -> $gs_uri"
  gsutil -m -o "GSUtil:parallel_composite_upload_threshold=150M" cp -n "$DEST_PATH" "$gs_uri"
fi

echo "[ok] $DEST_PATH"
