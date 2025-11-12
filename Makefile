# =========================
# MANTRA — Repro Makefile
# =========================

PROJECT ?= mantra-477901
ZONE    ?= us-west4-a
VM      ?= mantra-g2
BUCKET  ?= mantra-mlfg-prod-uscentral1-8e7a

SHELL := /bin/bash
GCLOUD ?= gcloud
REMOTE = $(VM) --project=$(PROJECT) --zone=$(ZONE)

# ---- Runners ----
# Local runner (you said you’re using pip+venv locally)
LOCAL_PY ?= python

# VM runner (force the conda env named "venv")
VM_CONDA_RUN = conda run -n venv
VM_PY        = $(VM_CONDA_RUN) python

.PHONY: help
help:
	@echo "Targets:"
	@echo "  bootstrap        : Upload and run tools/bootstrap_vm.sh on VM"
	@echo "  bootstrap.run    : Re-run bootstrap on VM (script already there)"
	@echo "  gcs.init         : Ensure GCS prefixes (data/raw, data/interim, manifests)"
	@echo "  gs.ls            : List bucket top-level and common prefixes"
	@echo "  vm.ssh           : Open interactive SSH to VM"
	@echo "  vm.cuda          : Check nvidia-smi & PyTorch CUDA on VM (uses conda env 'venv')"
	@echo "  vm.run           : Run pipeline target on VM (PIPELINE=<target>, default: all; enforces 'venv')"
	@echo "  all,data,qc,cnmf,theta,baseline : Local pipeline (uses LOCAL_PY=$(LOCAL_PY))"

# -------------------------
# VM bootstrap
# -------------------------
.PHONY: bootstrap bootstrap.run
bootstrap:
	$(GCLOUD) compute scp tools/bootstrap_vm.sh "$(VM):~/bootstrap_vm.sh" \
	  --project="$(PROJECT)" --zone="$(ZONE)"
	$(GCLOUD) compute ssh $(REMOTE) -- \
	  'bash -lc "chmod +x ~/bootstrap_vm.sh && PROJECT=$(PROJECT) BUCKET=$(BUCKET) ~/bootstrap_vm.sh"'

bootstrap.run:
	$(GCLOUD) compute ssh $(REMOTE) -- \
	  'bash -lc "PROJECT=$(PROJECT) BUCKET=$(BUCKET) ~/bootstrap_vm.sh"'

# -------------------------
# GCS helpers
# -------------------------
.PHONY: gcs.init gs.ls
gcs.init:
	$(GCLOUD) compute scp tools/init_gcs.sh "$(VM):~/init_gcs.sh" \
	  --project="$(PROJECT)" --zone="$(ZONE)"
	$(GCLOUD) compute ssh $(REMOTE) -- \
	  'bash -lc "chmod +x ~/init_gcs.sh && BUCKET=$(BUCKET) ~/init_gcs.sh"'

gs.ls:
	$(GCLOUD) compute ssh $(REMOTE) -- \
	  'bash -lc "gsutil ls gs://$(BUCKET)/; gsutil ls gs://$(BUCKET)/data/ || true; gsutil ls gs://$(BUCKET)/manifests/ || true"'

# -------------------------
# VM utilities
# -------------------------
.PHONY: vm.ssh vm.cuda vm.run
vm.ssh:
	$(GCLOUD) compute ssh $(REMOTE)

vm.cuda:
	$(GCLOUD) compute ssh $(REMOTE) -- 'bash -lc "\
		PY=\"$(VM_PY)\"; \
		echo \"== nvidia-smi ==\"; (nvidia-smi || sudo nvidia-smi || true); \
		echo \"== PyTorch CUDA check ==\"; \
		$$PY - <<PY \
import torch, sys; \
print(\"torch:\", getattr(torch, \"__version__\", \"?\")); \
print(\"cuda available:\", torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no gpu\"); \
PY"'

# Which pipeline target to run on the VM (reuses this Makefile on the VM)
PIPELINE ?= all
vm.run:
	$(GCLOUD) compute ssh $(REMOTE) -- 'bash -lc "\
		cd ~/MANTRA && \
		$(MAKE) $(PIPELINE) PY_CMD=\"$(VM_PY)\" "'

# -------------------------
# LOCAL pipeline (uses LOCAL_PY by default)
# You can override the runner: `make all PY_CMD="conda run -n venv python"`
# -------------------------
PY_CMD ?= $(LOCAL_PY)

.PHONY: all data qc cnmf theta baseline
all: data qc cnmf theta baseline

data:
	$(PY_CMD) scripts/00_fetch_data.py --config configs/paths.yml --manifest out/interim/manifest_data.json

qc:
	$(PY_CMD) scripts/01_qc_eda.py --params configs/params.yml --out out/interim --adata data/raw/K562/essential/K562_essential_normalized_singlecell_01.h5ad

cnmf:
	$(PY_CMD) scripts/02_cnmf.py --params configs/params.yml --in out/interim --out out/interim

theta:
	$(PY_CMD) scripts/03_fit_theta_wls.py --trait MCH --in out/interim --smr data/smr --out out/interim

baseline:
	$(PY_CMD) scripts/04_deltaE_to_trait.py --in out/interim --gwps data/gwps --out out/interim
	$(PY_CMD) scripts/05_metrics_and_plots.py --in out/interim --out out/interim

# -------------------------
# Data download to GCS from VM
# -------------------------
.PHONY: data.download
data.download:
	$(GCLOUD) compute scp tools/download_data.sh "$(VM):~/download_data.sh" \
	  --project="$(PROJECT)" --zone="$(ZONE)"
	$(GCLOUD) compute ssh $(REMOTE) -- 'bash -lc "\
		chmod +x ~/download_data.sh && \
		BUCKET='$(BUCKET)' \
		PREFIX=data/raw \
		MANIFEST=\$$HOME/MANTRA/configs/download_manifest.csv \
		WORKDIR=\$$HOME/tmp_downloads \
		~/download_data.sh"'

.PHONY: vm.fig.get
vm.fig.get:
	$(GCLOUD) compute scp tools/dl_fig.sh "$(VM):~/dl_fig.sh" \
	  --project="$(PROJECT)" --zone="$(ZONE)"
	$(GCLOUD) compute ssh $(REMOTE) -- 'bash -lc "\
	  chmod +x ~/dl_fig.sh && \
	  OUT=\$$HOME/MANTRA/data/raw && \
	  mkdir -p $$OUT && \
	  ~/dl_fig.sh \"$(URL)\" $$OUT gs://$(BUCKET)/data/raw"'


vm.ensure_dirs:
	@$(GCLOUD) compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE) -- \
	  'bash -lc "mkdir -p ~/MANTRA/data/raw ~/MANTRA/data/interim ~/MANTRA/data/smr ~/MANTRA/out/interim && echo [ok] ensured local workspace dirs"'

cleanup.staging:
	@$(GCLOUD) compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE) -- \
	  'bash -lc "cd ~/MANTRA && ./tools/cleanup.sh staging"'

cleanup.raw.local:
	@$(GCLOUD) compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE) -- \
	  'bash -lc "cd ~/MANTRA && ./tools/cleanup.sh raw-local"'

cleanup.all:
	@$(GCLOUD) compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE) -- \
	  'bash -lc "cd ~/MANTRA && ./tools/cleanup.sh all"'

.PHONY: sync.raw.down sync.raw.verify sync.interim.up sync.interim.down sync.out.up sync.smr.down sync.smr.up sync.all.up sync.all.down
sync.%:
	@$(GCLOUD) compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE) -- \
	  'bash -lc "cd ~/MANTRA && ./tools/sync.sh $*"'
