#!/usr/bin/env bash
set -euo pipefail

# ---- config (override via env) ----
: "${BUCKET:=mantra-mlfg-prod-uscentral1-8e7a}"
: "${RAW_DIR:=data/raw}"
: "${INTERIM_DIR:=data/interim}"
: "${OUT_INTERIM_DIR:=out/interim}"
: "${SMR_DIR:=data/smr}"

usage() {
  cat <<'USAGE'
Usage:
  sync.sh raw.down        # GCS → VM (cp -n) for immutable raw data
  sync.sh raw.verify      # compare sizes VM vs GCS for raw
  sync.sh interim.up      # VM → GCS (rsync -d) for interim artifacts
  sync.sh interim.down    # GCS → VM (rsync) if you ever need to pull back
  sync.sh out.up          # VM → GCS (rsync -d) for out/interim
  sync.sh smr.down        # GCS → VM for SMR/TWAS inputs (if stored in GCS)
  sync.sh smr.up          # VM → GCS for SMR (if VM is canonical)
  sync.sh all.up          # push interim + out/interim
  sync.sh all.down        # pull raw + smr
Env:
  BUCKET=mantra-mlfg-prod-uscentral1-8e7a
  RAW_DIR=data/raw         INTERIM_DIR=data/interim
  OUT_INTERIM_DIR=out/interim  SMR_DIR=data/smr
USAGE
}

ensure_dirs() {
  mkdir -p "$RAW_DIR" "$INTERIM_DIR" "$OUT_INTERIM_DIR" "$SMR_DIR"
}

raw_down() {
  ensure_dirs
  # Faster & simpler for immutable big files than rsync’s metadata crawl
  gsutil -m cp -n -r "gs://${BUCKET}/data/raw/*" "$RAW_DIR/"
}

raw_verify() {
  echo "== local raw =="
  du -sh "$RAW_DIR" || true
  echo "== gcs raw (sizes) =="
  gsutil ls -l "gs://${BUCKET}/data/raw/**" 2>/dev/null | awk 'NF==3{bytes+=$1} END{printf("Total: %.2f GiB\n", bytes/1024/1024/1024)}'
}

interim_up() {
  ensure_dirs
  gsutil -m rsync -r -d "$INTERIM_DIR/" "gs://${BUCKET}/data/interim/"
}

interim_down() {
  ensure_dirs
  gsutil -m rsync -r "gs://${BUCKET}/data/interim/" "$INTERIM_DIR/"
}

out_up() {
  ensure_dirs
  gsutil -m rsync -r -d "$OUT_INTERIM_DIR/" "gs://${BUCKET}/out/interim/"
}

smr_down() {
  ensure_dirs
  gsutil -m rsync -r "gs://${BUCKET}/data/smr/" "$SMR_DIR/" || true
}

smr_up() {
  ensure_dirs
  gsutil -m rsync -r -d "$SMR_DIR/" "gs://${BUCKET}/data/smr/"
}

case "${1:-}" in
  raw.down)      raw_down ;;
  raw.verify)    raw_verify ;;
  interim.up)    interim_up ;;
  interim.down)  interim_down ;;
  out.up)        out_up ;;
  smr.down)      smr_down ;;
  smr.up)        smr_up ;;
  all.up)        interim_up; out_up ;;
  all.down)      raw_down; smr_down ;;
  *)             usage; exit 2 ;;
esac
