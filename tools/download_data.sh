#!/usr/bin/env bash
set -euo pipefail

# ---- config (override via env or .env) ----
: "${BUCKET:=mantra-mlfg-prod-uscentral1-8e7a}"   # gs bucket name (no gs:// prefix)
: "${PREFIX:=data/raw}"                            # bucket subdir to place files under
: "${MANIFEST:=configs/download_manifest.csv}"     # csv with: url,dst_relpath[,sha256]
: "${WORKDIR:=/tmp/mantra_downloads}"              # tmp local staging dir
: "${CONCURRENCY:=4}"                              # parallel uploads (gsutil -m)
: "${RETRIES:=20}"                                 # network retry attempts

# Optional: pick downloader (aria2c if installed, else curl)
DOWNLOADER=""
# --- replace with this ---
if command -v aria2c >/dev/null 2>&1; then
  # single connection, robust resume; figshare presigned redirects can be touchy with multi-part
  DOWNLOADER="aria2c --check-integrity=false --continue=true -x1 -s1 --retry-wait=3 --max-tries=${RETRIES} -d \"${WORKDIR}\" -o"
else
  DOWNLOADER="curl -L --fail --retry ${RETRIES} --retry-all-errors --retry-delay 3 -C - -o"
fi


mkdir -p "${WORKDIR}"
cd "${WORKDIR}"
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

download_file() {
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
  filename="$(basename "${rel}")"

  # shellcheck disable=SC2086
  if ! eval ${DOWNLOADER} "\"${filename}\"" "\"${url}\""; then
    echo "   aria2/curl primary attempt failed; retrying once with curl single-stream..."
    curl -L --retry ${RETRIES} --retry-connrefused --retry-delay 3 -C - -o "${dest_local}" "${url}"
  fi
  # Return the local path on stdout so caller can capture it.
  echo "${dest_local}"

}

upload_file() {
  local rel="$1"
  local dest_local="$2"

  local dest_gs="gs://${BUCKET}/${PREFIX}/${rel}"

  # Clean up any 0-byte placeholder in GCS (from prior failed stream) so -n won't skip it.
  if gsutil stat "${dest_gs}" >/dev/null 2>&1; then
    local gsize
    gsize=$(gsutil stat "${dest_gs}" | awk '/Content-Length:/ {print $3}')
    if [[ "${gsize:-}" == "0" ]]; then
      echo "   found 0-byte object at ${dest_gs}; removing so we can reupload"
      gsutil rm -f "${dest_gs}" || true
    fi
  fi

  # Sanity: require a non-empty local file before uploading
  if [[ ! -s "${dest_local}" ]]; then
    echo "!! local file missing or empty: ${dest_local}" >&2
    return 1
  fi

  echo "==> Upload: ${dest_local} -> ${dest_gs}"
  gsutil "${GSUTIL_OPTS[@]}" cp -n "${dest_local}" "${dest_gs}"
  echo "==> Done: ${rel}"
}

# --- drive the rows (sequential; bump with xargs -P if you want later) ---
for line in "${LINES[@]}"; do
  IFS=',' read -r URL REL SHA <<<"${line}"
  local_path="$(download_file "${URL}" "${REL}" "${SHA:-}")"
  upload_file "${REL}" "${local_path}"
done

echo
echo "All done. Listing uploaded objects under gs://${BUCKET}/${PREFIX}:"
gsutil ls "gs://${BUCKET}/${PREFIX}/" || true
