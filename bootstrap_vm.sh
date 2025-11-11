#!/usr/bin/env bash
set -euo pipefail

log() { printf "\n[%s] %s\n" "$(date +'%F %T')" "$*"; }

# -------- 0) GPU sanity (driver shows up) --------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  # sometimes only sudo path has it
  if ! sudo -n true 2>/dev/null; then
    log "nvidia-smi not in PATH and no sudo; will try after driver path. Continuing..."
  fi
fi
sudo nvidia-smi || nvidia-smi || true

# -------- 1) Miniconda install (idempotent) --------
if [ ! -d "$HOME/miniconda3" ]; then
  log "Installing Miniconda..."
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$HOME/miniconda.sh"
  bash "$HOME/miniconda.sh" -b -p "$HOME/miniconda3"
  rm -f "$HOME/miniconda.sh"
  "$HOME/miniconda3/bin/conda" init bash || true
fi

# Load conda for THIS shell
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
fi

# -------- 2) Accept Anaconda ToS once (idempotent) --------
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true

# -------- 3) Get your repo (idempotent) --------
if [ ! -d "$HOME/MANTRA/.git" ]; then
  log "Cloning MANTRA..."
  git clone https://github.com/pvd232/MANTRA.git "$HOME/MANTRA"
else
  log "Repo exists; updating..."
  git -C "$HOME/MANTRA" fetch --all --prune
  git -C "$HOME/MANTRA" reset --hard origin/main
fi

# -------- 4) Build/Update env (no heredocs) --------
log "Creating/Updating conda env from configs/env.yml..."
ENV_FILE="$HOME/MANTRA/configs/env.yml"
if ! [ -f "$ENV_FILE" ]; then
  log "ERROR: $ENV_FILE not found"; exit 1
fi

# Normalize line endings (safe & idempotent)
sed -i 's/\r$//' "$ENV_FILE"

# Try create, then fall back to update
conda env create -f "$ENV_FILE" || conda env update -f "$ENV_FILE" --prune

# -------- 5) Activate & verify CUDA --------
log "Activating env 'mantra' and checking PyTorch/CUDA..."
conda activate mantra

python - <<'PY'
import torch, sys
print("Torch version:", getattr(torch, "__version__", "unknown"))
cuda = torch.cuda.is_available()
print("CUDA available:", cuda)
if cuda:
    print("Device count:", torch.cuda.device_count())
    print("Device 0:", torch.cuda.get_device_name(0))
else:
    sys.exit(2)  # helps surface "no CUDA" clearly
PY

# Driver sanity (again) — not fatal if needs sudo
sudo nvidia-smi || nvidia-smi || true

log "✅ Bootstrap complete."
