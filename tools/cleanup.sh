#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via env or flags)
: "${STAGE:=/home/machina/tmp_downloads}"
: "${RAW_ROOT:=/home/machina/MANTRA/data/raw}"

usage() {
  cat <<'USAGE'
Usage:
  cleanup.sh staging      # remove staging download dir on the VM
  cleanup.sh raw-local    # remove local RAW cache within repo on the VM
  cleanup.sh all          # do both
Env overrides:
  STAGE=/path/to/stage        (default: /home/machina/tmp_downloads)
  RAW_ROOT=/path/to/raw/root  (default: /home/machina/MANTRA/data/raw)
USAGE
}

cleanup_staging() {
  if [[ -d "$STAGE" ]]; then
    echo "[cleanup.staging] removing $STAGE"
    rm -rf "$STAGE"
  else
    echo "[cleanup.staging] nothing to remove (missing $STAGE)"
  fi
}

cleanup_raw_local() {
  if [[ -d "$RAW_ROOT" ]]; then
    echo "[cleanup.raw.local] removing $RAW_ROOT/*"
    rm -rf "$RAW_ROOT"/*
  else
    echo "[cleanup.raw.local] nothing to remove (missing $RAW_ROOT)"
  fi
}

[[ $# -ge 1 ]] || { usage; exit 2; }
case "$1" in
  staging)   cleanup_staging ;;
  raw-local) cleanup_raw_local ;;
  all)       cleanup_staging; cleanup_raw_local ;;
  *)         usage; exit 2 ;;
esac

echo
df -h /
