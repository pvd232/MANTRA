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

.PHONY: help
help:
	@echo "Targets:"
	@echo "  bootstrap        : Upload and run tools/bootstrap_vm.sh on VM"
	@echo "  bootstrap.run    : Re-run bootstrap on VM (script already there)"
	@echo "  gcs.init         : Ensure GCS prefixes (data/raw, data/interim, manifests)"
	@echo "  gs.ls            : List bucket top-level and common prefixes"
	@echo "  vm.ssh           : Open interactive SSH to VM"
	@echo "  vm.cuda          : Check nvidia-smi & PyTorch CUDA on VM"
	@echo "  vm.run           : Run pipeline target on VM (PIPELINE=<target>, default: all)"
	@echo "  all,data,qc,cnmf,theta,baseline : Local pipeline"

# -------------------------
# VM bootstrap
# -------------------------
.PHONY: bootstrap bootstrap.run
bootstrap:
	gcloud compute scp tools/bootstrap_vm.sh "$(VM):~/bootstrap_vm.sh" \
	  --project="$(PROJECT)" --zone="$(ZONE)"
	gcloud compute ssh $(REMOTE) -- \
	  'bash -lc "chmod +x ~/bootstrap_vm.sh && PROJECT=$(PROJECT) BUCKET=$(BUCKET) ~/bootstrap_vm.sh"'

bootstrap.run:
	gcloud compute ssh $(REMOTE) -- \
	  'bash -lc "PROJECT=$(PROJECT) BUCKET=$(BUCKET) ~/bootstrap_vm.sh"'

# -------------------------
# GCS helpers
# -------------------------
.PHONY: gcs.init gs.ls
gcs.init:
	gcloud compute scp tools/init_gcs.sh "$(VM):~/init_gcs.sh" \
	  --project="$(PROJECT)" --zone="$(ZONE)"
	gcloud compute ssh $(REMOTE) -- \
	  'bash -lc "chmod +x ~/init_gcs.sh && BUCKET=$(BUCKET) ~/init_gcs.sh"'

gs.ls:
	gcloud compute ssh $(REMOTE) -- \
	  'bash -lc "gsutil ls gs://$(BUCKET)/; gsutil ls gs://$(BUCKET)/data/ || true; gsutil ls gs://$(BUCKET)/manifests/ || true"'

# -------------------------
# VM utilities
# -------------------------
.PHONY: vm.ssh vm.cuda vm.run
vm.ssh:
	gcloud compute ssh $(REMOTE)

vm.cuda:
	gcloud compute ssh $(REMOTE) -- 'bash -lc "\
		source ~/miniconda3/etc/profile.d/conda.sh && conda activate mantra && \
		echo \"== nvidia-smi ==\" && (nvidia-smi || sudo nvidia-smi || true) && \
		echo \"== PyTorch CUDA check ==\" && python - <<PY \
import torch; print(\"torch:\", torch.__version__); print(\"cuda available:\", torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no gpu\"); \
PY"'

PIPELINE ?= all
vm.run:
	gcloud compute ssh $(REMOTE) -- 'bash -lc "\
		source ~/miniconda3/etc/profile.d/conda.sh && conda activate mantra && \
		cd ~/MANTRA && make $(PIPELINE)"'

# -------------------------
# LOCAL pipeline
# -------------------------
.PHONY: all data qc cnmf theta baseline

all: data qc cnmf theta baseline

data:
	python scripts/00_fetch_data.py --config configs/paths.yml --manifest out/interim/manifest_data.json

qc:
	python scripts/01_qc_eda.py --params configs/params.yml --out out/interim --adata data/raw/K562/essential/K562_essential_normalized_singlecell_01.h5ad

cnmf:
	python scripts/02_cnmf.py --params configs/params.yml --in out/interim --out out/interim

theta:
	python scripts/03_fit_theta_wls.py --trait MCH --in out/interim --smr data/smr --out out/interim

baseline:
	python scripts/04_deltaE_to_trait.py --in out/interim --gwps data/gwps --out out/interim
	python scripts/05_metrics_and_plots.py --in out/interim --out out/interim

# -------------------------
# Data download to GCS from VM
# -------------------------
.PHONY: data.download
data.download:
	gcloud compute scp tools/download_data.sh "$(VM):~/download_data.sh" \
	  --project="$(PROJECT)" --zone="$(ZONE)"
	gcloud compute ssh $(REMOTE) -- 'bash -lc "\
		chmod +x ~/download_data.sh && \
		BUCKET='$(BUCKET)' \
		PREFIX=data/raw \
		MANIFEST=\$$HOME/MANTRA/configs/download_manifest.csv \
		WORKDIR=\$$HOME/tmp_downloads \
		~/download_data.sh"'

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