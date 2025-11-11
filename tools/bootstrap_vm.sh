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

# assumes you've already created the 'mantra' conda env
set -euo pipefail

# Ensure we’re using conda’s shell functions (works in non-interactive scripts)
if command -v conda >/dev/null 2>&1; then
  eval "$(/usr/bin/env conda shell.bash hook)" || true
fi

# Prefer running inside the env via 'conda run' (avoids activation edge cases)
PYBIN="conda run -n mantra python"
if ! conda env list | awk '{print $1}' | grep -qx mantra; then
  # fall back to current python if env isn't present yet
  PYBIN="python"
fi

echo "Using interpreter: $PYBIN"
$PYBIN - <<'PY'
import sys, importlib, json
mods = ["scanpy","numpy","scipy","pandas"]
failures = {}
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "unknown")
        print(f"[OK] {m} {v}")
    except Exception as e:
        failures[m] = repr(e)
print("Python:", sys.version)
if failures:
    print("[ERROR] Import failures:", json.dumps(failures, indent=2))
    raise SystemExit(1)
PY
echo "Python + core libs imported successfully."


# Normalize line endings (safe & idempotent)
sed -i 's/\r$//' "$ENV_FILE"

# Try create, then fall back to update
conda env create -f "$ENV_FILE" || conda env update -f "$ENV_FILE" --prune

# -------- 5) Activate & verify CUDA --------

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-INTEL}"

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

# -------- 6) System deps for data downloads --------
# aria2 speeds up/robustifies large Figshare downloads (fallbacks to curl if absent)
if ! command -v aria2c >/dev/null 2>&1; then
  sudo apt-get update -y && sudo apt-get install -y aria2
fi

log "✅ Bootstrap complete."
