# MANTRA — Manifold-Aware Network TRAit modeling

> Entry point for an ML4FG project that couples a GRN prior (GWPS/Perturb-seq β), an optional manifold constraint, cNMF programs (W), and SMR/TWAS-based trait readout (theta). Interim scope: download + QC data, run EDA, fit cNMF, fit theta via weighted least squares (WLS), and produce a first DeltaE→DeltaTrait baseline with metrics and an ablation plan.

---

## TL;DR deliverables (interim)

- Data downloaded & QC’d (UMIs, %mito, HVGs).
- Baselines: cNMF (W) on unperturbed; theta^(t) via WLS from SMR; DeltaE→DeltaTrait baseline + metrics (R², AUROC/AUPRC, calibration).
- Ablation plan (2×2: GRN × Manifold) + compute budget.

---

## Project map

```
data/
  raw/                      # immutable sources (checksummed)
  interim/                  # QC’d AnnData + HVGs
  smr/                      # SMR/TWAS slices (per trait)
out/
  interim/
    qc_summary.csv
    qc_violin.png, hvg.png
    program_loadings_W.csv
    theta_<trait>.csv
    metrics_baseline_<trait>.csv
    calibration_curve_<trait>.tsv
configs/
  env.yml                   # conda env (locked)
  paths.yml                 # URLs/DOIs for data sources
  params.yml                # HVG count, K (programs), mu-grid, seeds
scripts/
  00_fetch_data.py
  01_qc_eda.py
  02_cnmf.py
  03_fit_theta_wls.py
  04_deltaE_to_trait.py
  05_metrics_and_plots.py
Makefile
mypy.ini
```

---

## Reproducibility protocol

- **Provenance & immutability.** Record every source in `data/RAW_SOURCES.md` with URL/DOI and SHA-256. Primary sources: GWPS/Perturb-seq (K562), HCT116 dose strata, and curated RBC-trait summaries for SMR/TWAS.
- **Configs over code.** Tunables live in `configs/params.yml` (HVGs, number of programs K, mu-grid, random seeds).
- **Determinism.** Fixed seeds; NMF uses `nndsvda` initialization.
- **Environment lock.** `configs/env.yml` (plus a `pip freeze` snapshot).
- **Logging.** Each script writes a manifest JSON: inputs → outputs, SHA-256, git SHA, wall-time, seeds.
- **Predeclared ablations.** 2×2 grid (GRN × Manifold); manifold smoother `(I + mu * L_M)^(-1) * DeltaE`.

---

## Methods (one screen)

1) **GRN prior.** Predict expression change with a GRN: `DeltaE = beta * u` (from GWPS).
2) **Manifold constraint (optional).** Learn geometry on unperturbed cells (e.g., EGGFM), build a Laplacian `L_M`, and smooth: `DeltaE_tilde = (I + mu * L_M)^(-1) * DeltaE`.
3) **Program mapping.** Map to programs from cNMF: `Delta_a = W^T * DeltaE_tilde`.
4) **Trait readout (SMR → theta).** Fit a weighted least-squares model of SMR gene effects onto the program matrix `W` to obtain `theta^(t)`.

**Prediction:** `DeltaTrait_hat^(t) = dot(theta^(t), Delta_a)`.

---

## Ablations (fixed)

1) **No-GRN / No-Manifold** — program-only baseline  
2) **GRN / No-Manifold** — current interim baseline  
3) **No-GRN / Manifold** — diffuse a data-only DeltaE  
4) **GRN / Manifold** — apply manifold smoothing before `W` → `theta`

---

## Metrics scaffold

- **Regression:** R²  
- **Correlation:** Pearson and Spearman  
- **Calibration:** slope, intercept, and Brier score  
- **Sign prediction:** AUROC and AUPRC  
- **(Optional) Manifold realism:** kNN-overlap and geodesic-path change

---

## Quickstart

```bash
conda env create -f configs/env.yml
conda activate mantra
make all  # or run targets below
```

Common targets:

```bash
make data     # fetch raw files and write a manifest
make qc       # QC + EDA → out/interim/*
make cnmf     # fit cNMF → program_loadings_W.csv
make theta    # fit theta via WLS → theta_<trait>.csv
make baseline # predict + metrics → yhat & metrics files
```

---
Update VM git repo:
```bash
gcloud compute ssh mantra-g2 --project=mantra-477901 --zone=us-west4-a -- \                                         
  'bash -lc "
    cd ~/MANTRA
    git pull
  "'
```


## Data sources

- **GWPS (K562/HCT116):** for beta (regulator→gene effects) and dose strata (Q1–Q4).
- **Causal genetics (UKB RBC traits):** SMR/TWAS slices used to fit `theta^(t)`.

See `data/RAW_SOURCES.md` for exact endpoints and checksums.

---

## License / credits

If you use this code or ideas, please cite the relevant datasets and methods (e.g., GWPS/Perturb-seq sources, SMR/TWAS, and any geometry-learning method used).
