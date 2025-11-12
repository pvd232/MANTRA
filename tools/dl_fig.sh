#!/usr/bin/env bash
set -euo pipefail
# usage: dl_fig.sh <figshare_download_url> <outdir> [gs://bucket/prefix/]
URL="${1:?download URL required}"; OUT="${2:-.}"; GSU="${3:-}"
mkdir -p "$OUT"
name="$(basename "${URL%%\?*}")"
dest="$OUT/$name"
if command -v aria2c >/dev/null; then
  aria2c -c -x8 -s8 -d "$OUT" -o "$name" "$URL"
else
  curl -fL --retry 5 --retry-delay 2 -o "$dest" "$URL"
fi
[ -n "${GSU}" ] && gsutil cp -n "$dest" "${GSU%/}/"
echo "[ok] $dest"