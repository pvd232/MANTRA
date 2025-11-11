.PHONY: all data qc cnmf theta baseline

all: data qc cnmf theta baseline

data:
	python scripts/00_fetch_data.py --config configs/paths.yml --manifest out/interim/manifest_data.json

qc:
	python scripts/01_qc_eda.py --params configs/params.yml --out out/interim --adata data/interim/unperturbed.h5ad

cnmf:
	python scripts/02_cnmf.py --params configs/params.yml --in out/interim --out out/interim

theta:
	python scripts/03_fit_theta_wls.py --trait MCH --in out/interim --smr data/smr --out out/interim

baseline:
	python scripts/04_deltaE_to_trait.py --in out/interim --gwps data/gwps --out out/interim
	python scripts/05_metrics_and_plots.py --in out/interim --out out/interim
