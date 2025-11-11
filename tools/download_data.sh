#!/usr/bin/env bash
set -euo pipefail

# ---- config (override via env or .env) ----
: "${BUCKET:=mantra-mlfg-prod-uscentral1-8e7a}"   # gs bucket name (no gs:// prefix)
: "${PREFIX:=data/raw}"                            # bucket subdir to place files under
: "${MANIFEST:=configs/download_manifest.csv}"     # csv with: url,dst_relpath[,sha256]
: "${WORKDIR:=/tmp/mantra_downloads}"              # tmp local staging dir
: "${CONCURRENCY:=4}"                              # parallel uploads (gsutil -m)
: "${RETRIES:=10}"                                 # network retry attempts

# Optional: pick downloader (aria2c if installed, else curl)
DOWNLOADER=""
if command -v aria2c >/dev/null 2>&1; then
  DOWNLOADER="aria2c --check-integrity=false --continue=true -x16 -s16 --retry-wait=2 --max-tries=${RETRIES} -d \"${WORKDIR}\" -o"
else
  DOWNLOADER="curl -L --retry ${RETRIES} --retry-connrefused --retry-delay 2 -C - -o"
fi

mkdir -p "${WORKDIR}"

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "fatal: missing dependency: $1" >&2; exit 1; }
}

# deps we use
require gsutil
require python3
require awk
require sed

# sanity: manifest exists?
[[ -f "${MANIFEST}" ]] || { echo "fatal: manifest not found: ${MANIFEST}" >&2; exit 1; }

echo "Using bucket: gs://${BUCKET}"
echo "Manifest: ${MANIFEST}"
echo "Staging to: ${WORKDIR}"
echo

# Enable fast parallel uploads for large files
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
GSUTIL_OPTS=(-m -o "GSUtil:parallel_composite_upload_threshold=150M")

# parse CSV lines: url,dst_relpath[,sha256]
# - skips comments (# ...) and blank lines
# - trims whitespace
mapfile -t LINES < <(awk -F, '
  BEGIN{OFS=","}
  /^[[:space:]]*$/ {next}
  /^[[:space:]]*#/ {next}
  {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1);
   gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2);
   gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3);
   print $1","$2","$3}
' "${MANIFEST}")

download_one() {
  local url="$1"
  local rel="$2"
  local sha="$3"

  if [[ -z "${url}" || -z "${rel}" ]]; then
    echo "skip malformed line: '${url},${rel},${sha}'" >&2
    return 0
  fi

  local dest_local="${WORKDIR}/$(basename "${rel}")"
  local dest_gs="gs://${BUCKET}/${PREFIX}/${rel}"

  echo "==> Fetch: ${url}"
  # shellcheck disable=SC2086
  eval ${DOWNLOADER} "\"${dest_local}\"" "\"${url}\""

  if [[ -n "${sha}" ]]; then
    echo "==> Verify sha256: ${dest_local}"
    printf "%s  %s\n" "${sha}" "${dest_local}" | sha256sum -c -
  fi

  echo "==> Upload: ${dest_local} -> ${dest_gs}"
  gsutil "${GSUTIL_OPTS[@]}" cp -n "${dest_local}" "${dest_gs}"

  echo "==> Done: ${rel}"
}

# loop (sequential is simplest/reliable; bump to parallel with xargs -P if desired)
for line in "${LINES[@]}"; do
  IFS=',' read -r URL REL SHA <<<"${line}"
  # ensure subdirs exist in the bucket path (gsutil cp will create as needed)
  download_one "${URL}" "${REL}" "${SHA:-}"
done

echo
echo "All done. Listing uploaded objects under gs://${BUCKET}/${PREFIX}:"
gsutil ls "gs://${BUCKET}/${PREFIX}/" || true
