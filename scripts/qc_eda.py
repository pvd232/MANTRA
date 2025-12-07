#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore
from scipy import sparse
import subprocess
import yaml
import matplotlib.pyplot as plt
from dcol_pca import dcol_pca0, plot_spectral

# Example use
# conda run -n venv python scripts/01_qc_eda.py   --params configs/params.yml   --out out/interim   --ad data/raw/K562_gwps/k562_replogie.h5ad   --report   --report-to-gcs gs://mantra-mlfg-prod-uscentral1-8e7a/out/interim   --plot-max-cells 10000

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QC + EDA for unperturbed cells.")
    ap.add_argument("--params", required=True, help="configs/params.yml")
    ap.add_argument("--out", required=True, help="out/interim")
    ap.add_argument("--ad", required=True, help="path to unperturbed .h5ad")
    return ap

def prep(ad: sc.AnnData, params: Dict[str, Any]):
    n_cells = ad.n_obs

    # Remove genes that are not statistically relevant (< 0.1% of cells)
    min_cells = max(3, int(0.001 * n_cells))
    sc.pp.filter_genes(ad, min_cells=min_cells)

    # Remove empty droplets (cells with no detected genes)
    sc.pp.filter_cells(ad, min_genes=int(params["qc"]["min_genes"]))

    # Drop zero-count cells
    totals = np.ravel(ad.X.sum(axis=1))
    ad = ad[totals > 0, :].copy()

    # Cells with high percent of mitochondrial DNA are dying or damaged
    ad = ad[ad.obs["mitopercent"] < float(params["qc"]["max_pct_mt"])].copy()

    print("AnnData layers:", list(ad.layers.keys()), flush=True)
    print("AnnData obs columns:", list(ad.obs.columns), flush=True)
    print("AnnData var columns:", list(ad.var.columns), flush=True)

    # How many genes/cells remain just before HVG?
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

    # No raw counts object so we must use ad.X
    sc.pp.highly_variable_genes(
        ad,
        n_top_genes=int(params["hvg_n_top_genes"]),
        flavor="seurat_v3",
        subset=False,
    )

    ad = ad[:, ad.var["highly_variable"]].copy()

    # now normalize/log on X (leave counts in layer untouched)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    return ad


def main() -> None:
    args = build_argparser().parse_args()
    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load full AnnData in backed mode (no 61 GiB dense allocation) ---
    ad_full = sc.read_h5ad(args.ad, backed="r")
    print(
        f"[load] full AnnData: n_obs={ad_full.n_obs}, n_vars={ad_full.n_vars}",
        flush=True,
    )

    # --- Define unperturbed / control cells: gene == 'non-targeting' ---
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

    # --- Materialize ONLY the non-targeting cells in memory ---
    ad = ad_full[is_ctrl, :].to_memory()
    print(
        f"[load] using {ad.n_obs} non-targeting cells for QC + dim reduction",
        flush=True,
    )

    if not sparse.issparse(ad.X):
        ad.X = sparse.csr_matrix(ad.X)

    for col in ad.obs.columns:
        print(f"self.{col}: {ad.obs[col].dtype}", flush=True)

    print()
    for col in ad.var.columns:
        print(f"self.{col}: {ad.var[col].dtype}", flush=True)

    # QC processing
    qc_ad = prep(ad.copy(), params)
    print(f"[write] writing QC AnnData to {out_dir}", flush=True)
    
    qc_ad.write_h5ad(out_dir)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
