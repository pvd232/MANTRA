#!/usr/bin/env bash
set -euo pipefail

# ---------- Config (override via env) ----------
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
NAME_PREFIX="${NAME_PREFIX:-mantra}"
MACHINE="${MACHINE:-g2-standard-8}"             # for L4; override to n1-standard-8 for T4, a2-highgpu-1g for A100
GPU_TYPES=(${GPU_TYPES:-nvidia-l4 nvidia-tesla-t4 nvidia-tesla-a100})
GPU_COUNT="${GPU_COUNT:-1}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-7-cu128-ubuntu-2204-nvidia-570}"
BOOT_TYPE="${BOOT_TYPE:-pd-ssd}"
BOOT_SIZE_GB="${BOOT_SIZE_GB:-200}"
SCOPES="${SCOPES:-cloud-platform}"
SPOT="${SPOT:-0}"                                # set SPOT=1 for Spot/Preemptible
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"                # per attempt
VERBOSITY="${VERBOSITY:---verbosity=error}"      # or --verbosity=info/debug

# ---------- timeout/gtimeout shim ----------
if command -v timeout >/dev/null 2>&1; then TO=timeout
elif command -v gtimeout >/dev/null 2>&1; then TO=gtimeout
else TO=""; fi

fail() { echo "❌ $*"; exit 1; }

[[ -n "${PROJECT}" ]] || fail "No GCP project set (run: gcloud config set project <id>)."

echo "Project: ${PROJECT}"
echo "Image:   ${IMAGE_PROJECT}/${IMAGE_FAMILY}"
echo "Try GPUs in order: ${GPU_TYPES[*]}"
echo "Provisioning: $([[ $SPOT = 1 ]] && echo SPOT || echo ON_DEMAND)"
echo

# Fetch *all* zones once
mapfile -t ZONES < <(gcloud compute zones list --project="${PROJECT}" --format="value(name)" | sort)
[[ ${#ZONES[@]} -gt 0 ]] || fail "Could not list zones."

try_create() {
  local zone="$1" gputype="$2" name="$3"
  local pmflag=()
  [[ "$SPOT" = "1" ]] && pmflag=(--provisioning-model=SPOT)

  echo "→ Creating ${name} in ${zone} (${MACHINE}, ${GPU_COUNT}×${gputype}) …"
  if [[ -n "$TO" ]]; then
    $TO "${TIMEOUT_SEC}s" gcloud compute instances create "${name}" \
      --project="${PROJECT}" --zone="${zone}" \
      --machine-type="${MACHINE}" \
      --accelerator="count=${GPU_COUNT},type=${gputype}" \
      --maintenance-policy=TERMINATE \
      --image-project="${IMAGE_PROJECT}" --image-family="${IMAGE_FAMILY}" \
      --boot-disk-type="${BOOT_TYPE}" --boot-disk-size="${BOOT_SIZE_GB}" \
      --scopes="${SCOPES}" ${VERBOSITY} "${pmflag[@]}"
  else
    gcloud compute instances create "${name}" \
      --project="${PROJECT}" --zone="${zone}" \
      --machine-type="${MACHINE}" \
      --accelerator="count=${GPU_COUNT},type=${gputype}" \
      --maintenance-policy=TERMINATE \
      --image-project="${IMAGE_PROJECT}" --image-family="${IMAGE_FAMILY}" \
      --boot-disk-type="${BOOT_TYPE}" --boot-disk-size="${BOOT_SIZE_GB}" \
      --scopes="${SCOPES}" ${VERBOSITY} "${pmflag[@]}"
  fi
}

# Loop: for each GPU type, scan all zones; skip zones that don't offer that accelerator
idx=1
for gputype in "${GPU_TYPES[@]}"; do
  echo "=== Trying GPU: ${gputype} ==="
  for zone in "${ZONES[@]}"; do
    # quick capability check to avoid pointless API calls
    have=$(gcloud compute accelerator-types list \
             --project="${PROJECT}" --zones="${zone}" \
             --filter="name=${gputype}" --format="value(name)" 2>/dev/null || true)
    [[ -z "$have" ]] && { echo "  • ${zone}: ${gputype} not offered, skipping"; continue; }

    name="${NAME_PREFIX}-$(printf '%s-%03d' "${gputype//nvidia-/}" "$idx")"
    if try_create "${zone}" "${gputype}" "${name}"; then
      echo "✅ Created ${name} in ${zone}"
      echo "export VM=${name}"
      echo "export ZONE=${zone}"
      exit 0
    else
      echo "  ✗ ${zone} failed (quota/capacity or timeout)."
    fi
    idx=$((idx+1))
  done
done

fail "Failed to create a GPU VM in all zones for any of: ${GPU_TYPES[*]}"
