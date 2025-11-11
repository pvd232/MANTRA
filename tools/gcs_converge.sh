#!/usr/bin/env bash
set -euo pipefail

BUCKET=${BUCKET:-mantra-mlfg-prod-uscentral1-8e7a}
ROOT="gs://${BUCKET}"

echo "== current top-level =="
gsutil ls "${ROOT}/"

echo
echo "== DRY-RUN plan: move everything from /raw/* -> /data/raw/* =="
for p in K562 k562_perturbseq gtex_v8_eqtl metadata ukb_rbc_gwas figshare; do
  echo "Would move: ${ROOT}/raw/${p}/**  ->  ${ROOT}/data/raw/${p}/"
done

echo
read -p "Proceed with MOVEs? (yes/NO): " go
[[ "${go}" == "yes" ]] || { echo "Aborting."; exit 1; }

# Ensure destination prefixes exist
for p in data/raw data/interim data/smr out/interim manifests; do
  gsutil -q stat "${ROOT}/${p}/.keep" || echo "" | gsutil cp - "${ROOT}/${p}/.keep"
done

# Server-side renames (fast): raw/*  -> data/raw/*
for p in K562 k562_perturbseq gtex_v8_eqtl metadata ukb_rbc_gwas figshare; do
  src="${ROOT}/raw/${p}/**"
  dst="${ROOT}/data/raw/${p}/"
  echo "Moving ${src} -> ${dst}"
  gsutil -m mv "${src}" "${dst}" || true
done

# Optional: quarantine unknown extras under data/legacy/
echo
echo "== Quarantine unexpected directories under data/legacy/ (optional) =="
gsutil -m rsync -r "${ROOT}/data/gwps/"   "${ROOT}/data/legacy/gwps/"   && gsutil -m rm -r "${ROOT}/data/gwps/"   || true
gsutil -m rsync -r "${ROOT}/data/labels/" "${ROOT}/data/legacy/labels/" && gsutil -m rm -r "${ROOT}/data/labels/" || true
gsutil -m rsync -r "${ROOT}/data/external/" "${ROOT}/data/legacy/external/" && gsutil -m rm -r "${ROOT}/data/external/" || true

# Remove empty top-level raw/
echo
echo "== Clean empty raw/ prefix =="
gsutil -m rm -r "${ROOT}/raw" || true

echo
echo "== Final check =="
gsutil ls -r "${ROOT}/data/raw/**" | sed -n '1,200p'
