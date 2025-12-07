# MANTRA — Manifold-Aware Network TRAit modeling

> Entry point for an ML4FG project that couples a GRN prior (GWPS/Perturb-seq β), a learned cell-state manifold (EGGFM), cNMF programs (W), and SMR/TWAS-based trait readout (θ).  
> Current scope: construct robust cell-state geometry in K562, train EGGFM energy models on HVG subsets, integrate embeddings into GRN + GNN pipelines, and prepare ΔE→ΔTrait ablations.

---

## Current status (Dec 2025)

### **Data / preprocessing**

- K562 GWPS (unperturbed) is QC’d and stored in AnnData:
  - ~250k cells after stringent filtering.
  - 3,000 highly variable genes annotated.
- HVG subsets evaluated for EGGFM training stability:
  - {50, 75, 100, 150, 200, 250, 500}
  - Best DSM stability + smoothest loss curve achieved at **75 HVGs**.

### **Manifold / embeddings**

Stored in `K562_gwps_unperturbed_hvg_embeddings.h5ad`:

- `X_hvg_trunc` — raw expression of top **150** HVGs.
- `X_pca` — Scanpy PCA (20 PCs).
- `X_diffmap` — Diffusion Map (20 components).
- `X_umap` — UMAP (20D).
- `X_phate` — optional, only if PHATE is installed.
- `X_isomap` — optional; slow at this scale.
- `X_spectral` — not necessary since Diffusion Map already captures Laplacian geometry.

### **EGGFM / energy model**

- DSM-trained energy models using HVG subsets.
- **Best performing model**: **HVG=75**.
- Checkpoints saved as:
 - out/models/eggfm/eggfm_energy_k562_hvg75.pt
 - out/models/eggfm/eggfm_energy_k562_hvg<N>.pt # other ablations


Each checkpoint contains:
- `state_dict`  
- feature statistics  
- input gene names  
- architecture (hidden dims)  
- EGGFM feature space metadata  

### **Planned next steps**

1. Construct EGGFM-derived metric and diffusion operator.
2. Compare PCA / DiffMap / PHATE / EGGFM embeddings using ARI + TI stability.
3. Train GRN-aware GNN on regulator→gene graph.
4. Integrate manifold embeddings into ΔE→ΔTrait prediction.
5. Perform full GRN × Manifold ablation grid.

---

## TL;DR deliverables (interim)

- QC’d K562 unperturbed dataset with 3k HVGs.
- Multiple manifold embeddings on HVG-restricted space.
- EGGFM energy model trained on 75 HVGs.
- Ablation plan for embedding choice + GRN prior.
- Documentation for reproducibility, compute, VM usage.

---

## Project map

data/
raw/ # immutable sources, SHA-256 tracked
interim/
K562_gwps_unperturbed_qc.h5ad # QC’d dataset, 3k HVGs
K562_gwps_unperturbed_hvg_embeddings.h5ad
smr/ # SMR/TWAS slices per trait

out/
interim/
qc_summary.csv
qc_violin.png
hvg_rankplot.png
program_loadings_W.csv
theta_<trait>.csv
metrics_baseline_<trait>.csv
calibration_curve_<trait>.tsv
models/
eggfm/
eggfm_energy_k562_hvg75.pt
eggfm_energy_k562_hvg*.pt

configs/
env.yml
paths.yml
params.yml

scripts/
00_fetch_data.py
01_qc_eda.py
02_cnmf.py
03_fit_theta_wls.py
04_deltaE_to_trait.py
05_metrics_and_plots.py
hvg_embed.py
train_energy.py

Makefile


---

## Reproducibility protocol

### **Provenance & immutability**
- Every dataset logged in `data/RAW_SOURCES.md` with URL + SHA-256.

### **Configs over code**
Everything tunable is inside `configs/params.yml`:
- `hvg_total`, `max_hvg`
- EGGFM architecture
- DiffMap/UMAP/PHATE params
- seeds

### **Determinism**
- Fixed seeds for NumPy, PyTorch, Scanpy.
- Track any nondeterministic GPU ops.

### **Environment locking**
- `configs/env.yml` mirrors exact dependencies.
- `pip freeze` snapshot saved separately.

### **Manifest logging**
Each script writes a machine-readable manifest:
- Inputs → outputs
- SHA-256 hashes
- git SHA
- runtime
- hyperparams

### **Ablations**
**2×2 GRN × Manifold grid**:

| GRN | Manifold | Description |
|-----|----------|-------------|
| ✗ | ✗ | Program-only baseline |
| ✓ | ✗ | GRN-only ΔE→ΔTrait baseline |
| ✗ | ✓ | Geometry-only diffusion of ΔE |
| ✓ | ✓ | Full model: GRN + manifold |

**Embedding ablations**:
- Raw HVG-150
- PCA-20
- DiffMap-20
- UMAP-20
- PHATE-20 (optional)
- EGGFM-metric embedding (future milestone)

---

## Methods (current design)

### **1. GRN prior (GWPS β)**

\[
\Delta E_{\text{GRN}} = \beta u
\]

### **2. Manifold constraint**

Classical smoothing:

\[
\tilde{\Delta E} = (I + \mu L_M)^{-1} \Delta E
\]

EGGFM version (in progress):
- Define metric from energy model.
- Construct diffusion operator.
- Evaluate TI robustness + ARI.

### **3. Program mapping**

\[
\Delta a = W^\top \tilde{\Delta E}
\]

### **4. Trait readout (SMR→θ)**

\[
g^{(t)} \approx W \theta^{(t)}
\]

### **5. Prediction**

\[
\widehat{\Delta \text{Trait}}^{(t)} = \langle \theta^{(t)}, \Delta a \rangle
\]

---

## Metrics scaffold

- **R²**, Pearson, Spearman  
- Calibration slope, intercept, Brier  
- AUROC / AUPRC for sign prediction  
- Manifold realism:
  - kNN overlap
  - geodesic distortion

---

## Quickstart

```bash
conda env create -f configs/env.yml
conda activate venv
make data
make qc
