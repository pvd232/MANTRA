# src/mantra/qc.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import scanpy as sc  # type: ignore
from scipy import sparse
import yaml


def prep(ad: sc.AnnData, params: Dict[str, Any]) -> sc.AnnData:
    """
    Core QC + HVG selection + normalization on an in-memory AnnData.
    Returns a new AnnData containing only QC-passing cells + HVGs,
    log-normalized in ad.X, with raw counts saved in ad.layers['counts'].
    """
    n_cells = ad.n_obs

    # -------------------------
    # 1. Gene + cell filters
    # -------------------------
    # Use qc.min_cells if provided, else fallback to 0.1% of cells
    qc_cfg = params.get("qc", {})
    min_cells_cfg = int(qc_cfg.get("min_cells", 0))
    if min_cells_cfg > 0:
        min_cells = min_cells_cfg
    else:
        min_cells = max(3, int(0.001 * n_cells))

    print(f"[QC] filter_genes min_cells={min_cells}", flush=True)
    sc.pp.filter_genes(ad, min_cells=min_cells)

    sc.pp.filter_cells(ad, min_genes=int(qc_cfg["min_genes"]))

    # Drop zero-count cells
    totals = np.ravel(ad.X.sum(axis=1))
    ad = ad[totals > 0, :].copy()

    # Mito filter
    ad = ad[ad.obs["mitopercent"] < float(qc_cfg["max_pct_mt"])].copy()

    print("AnnData layers:", list(ad.layers.keys()), flush=True)
    print("AnnData obs columns:", list(ad.obs.columns), flush=True)
    print("AnnData var columns:", list(ad.var.columns), flush=True)
    print("n_obs, n_vars:", ad.n_obs, ad.n_vars, flush=True)

    # Check for inf/nan in means explicitly:
    X = ad.X
    if sparse.issparse(X):
        means = np.asarray(X.mean(axis=0)).ravel()
    else:
        means = np.nanmean(X, axis=0)

    print("Means finite?", np.all(np.isfinite(means)), flush=True)
    print("Means min/max:", np.nanmin(means), np.nanmax(means), flush=True)
    print("# non-finite means:", np.sum(~np.isfinite(means)), flush=True)

    # -------------------------
    # 2. Preserve raw counts
    # -------------------------
    # Save raw counts BEFORE HVG subset + normalization/log.
    # This will be written to disk as a 'counts' layer in the QC .h5ad.
    if "counts" not in ad.layers:
        # For large matrices, this is still sparse-aware if ad.X is sparse
        ad.layers["counts"] = ad.X.copy()
        print("[QC] Saved raw counts into ad.layers['counts']", flush=True)
    else:
        print("[QC] ad.layers['counts'] already present; not overwriting.", flush=True)

    # -------------------------
    # 3. HVG selection
    # -------------------------
    hvg_n_top = int(params["hvg_n_top_genes"])
    print(f"[QC] Computing HVGs with n_top_genes={hvg_n_top}", flush=True)

    sc.pp.highly_variable_genes(
        ad,
        n_top_genes=hvg_n_top,
        flavor="seurat_v3",
        subset=False,
    )

    n_hvg = int(ad.var["highly_variable"].sum())
    print(
        f"[QC] HVGs flagged={n_hvg} (requested n_top_genes={hvg_n_top}); "
        f"n_vars BEFORE subset={ad.n_vars}",
        flush=True,
    )

    ad = ad[:, ad.var["highly_variable"]].copy()
    print(f"[QC] n_vars AFTER HVG subset={ad.n_vars}", flush=True)

    # -------------------------
    # 4. Normalize / log-transform
    # -------------------------
    # Now normalize/log on X; raw counts are preserved in ad.layers['counts'].
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    return ad



def run_qc(
    params_path: Path,
    ad_path: Path,
    out_path: Path,
    *,
    pet: bool = False,
) -> sc.AnnData:
    """
    High-level QC entrypoint used by scripts / notebooks.

    params_path: YAML with 'qc' and 'hvg_n_top_genes' entries.
    ad_path:     Raw big .h5ad (possibly > 60GB).
    out_path:    Where to write QC’d .h5ad.
    pet:         If True, restrict to obs['gene'] == 'non-targeting'.
    """
    params: Dict[str, Any] = yaml.safe_load(params_path.read_text())

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load full AnnData in backed mode (no 61 GiB dense allocation) ---
    ad_full = sc.read_h5ad(str(ad_path), backed="r")
    print(
        f"[load] full AnnData: n_obs={ad_full.n_obs}, n_vars={ad_full.n_vars}",
        flush=True,
    )

    # Decide which cells to use
    if pet:
        # restrict to non-targeting controls in a full (perturbed+control) dataset
        if "gene" not in ad_full.obs:
            raise ValueError(
                "'gene' column not found in ad.obs. "
                f"Available columns: {list(ad_full.obs.columns)}"
            )

        is_ctrl = np.asarray(ad_full.obs["gene"] == "non-targeting")
        n_ctrl = int(is_ctrl.sum())
        n_pert = int((~is_ctrl).sum())
        print(f"[split] control/non-targeting cells: {n_ctrl}", flush=True)
        print(f"[split] perturbed cells: {n_pert}", flush=True)

        if n_ctrl == 0:
            raise ValueError("No control cells with gene == 'non-targeting' found.")

        ad = ad_full[is_ctrl, :].to_memory()
        print(
            f"[load] using {ad.n_obs} non-targeting cells for QC + dim reduction",
            flush=True,
        )
    else:
        # use all cells from the input .h5ad
        ad = ad_full.to_memory()
        print(
            f"[load] using all {ad.n_obs} cells for QC + dim reduction",
            flush=True,
        )

    # ensure sparse CSR for downstream ops
    from scipy import sparse as sp_sparse
    if not sp_sparse.issparse(ad.X):
        ad.X = sp_sparse.csr_matrix(ad.X)

    for col in ad.obs.columns:
        print(f"self.{col}: {ad.obs[col].dtype}", flush=True)

    print()
    for col in ad.var.columns:
        print(f"self.{col}: {ad.var[col].dtype}", flush=True)

    # QC processing
    qc_ad = prep(ad.copy(), params)

    print(f"[write] writing QC AnnData to {out_path}", flush=True)
    print("        QC ad.X shape:", qc_ad.X.shape, flush=True)
    if "counts" in qc_ad.layers:
        print("        QC counts layer shape:", qc_ad.layers["counts"].shape, flush=True)
    qc_ad.write_h5ad(out_path)

    return qc_ad
