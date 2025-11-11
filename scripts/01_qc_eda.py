#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, cast

import pandas as pd
import numpy as np
import scanpy as sc  # type: ignore
from scipy import sparse
import yaml


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QC + EDA for unperturbed cells.")
    ap.add_argument("--params", required=True, help="configs/params.yml")
    ap.add_argument("--out", required=True, help="out/interim")
    ap.add_argument("--adata", required=True, help="path to unperturbed .h5ad")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load AnnData
    ad = sc.read_h5ad(args.adata).copy()
    print("ad", ad.shape)

    # --- trust existing mitopercent (must live in ad.obs) ---
    if "mitopercent" not in ad.obs.columns:
        raise KeyError(
            f"'mitopercent' not found in ad.obs. Available obs columns: {list(ad.obs.columns)[:25]} ..."
        )

    # coerce to numeric and standardize scale to [0, 100]
    ad.obs["mitopercent"] = pd.to_numeric(ad.obs["mitopercent"], errors="coerce")
    if ad.obs["mitopercent"].isna().any():
        # drop rows where it's NaN (or fill if you prefer)
        ad = ad[~ad.obs["mitopercent"].isna(), :].copy()

    # if it's in [0,1], convert to %
    if ad.obs["mitopercent"].max() <= 1.0:
        ad.obs["mitopercent"] = 100.0 * ad.obs["mitopercent"]
    # Choose a count-like matrix
    if "counts" in (ad.layers or {}):
        X_counts = ad.layers["counts"]
        counts_src = "layers['counts']"
    elif ad.raw is not None:
        X_counts = ad.raw.X
        counts_src = "raw.X"
    else:
        X_counts = ad.X
        counts_src = "X (fallback)"

    print(f"[QC] Using {counts_src} as counts source")

    totals = np.ravel(X_counts.sum(axis=1))
    keep_nonzero = totals > 0
    if not keep_nonzero.all():
        print(
            f"[QC] Dropping {(~keep_nonzero).sum()} zero-count cells before normalize/log"
        )
    ad = ad[keep_nonzero, :].copy()

    # ---- QC metrics ----
    sc.pp.calculate_qc_metrics(
        ad,
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # Filters (explicit casts for Pylance)
    min_genes = int(params["min_genes_per_cell"])
    pct_mito_max = float(params["pct_mito_max"])

    mask = (ad.obs["n_genes_by_counts"] > min_genes) & (
        ad.obs["mitopercent"] < pct_mito_max
    )
    ad = ad[mask, :].copy()

    # Normalize + log + HVGs
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(
        ad, n_top_genes=int(params["hvg_n_top_genes"]), flavor="seurat_v3"
    )

    # Decide HVG flavor from counts, not from ad.X
    def is_integer_like_matrix(M) -> bool:
        data = M.data if sparse.issparse(M) else np.ravel(M)
        return (
            data.size > 0
            and np.all(np.isfinite(data))
            and np.allclose(data, np.round(data), atol=1e-8)
        )

    hvg_flavor = "seurat_v3" if is_integer_like_matrix(X_counts) else "seurat"
    print(f"[HVG] Using flavor={hvg_flavor}")

    sc.pp.highly_variable_genes(
        ad, n_top_genes=int(params["hvg_n_top_genes"]), flavor=hvg_flavor
    )

    # Persist QC’d AnnData for downstream steps
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    ad.write_h5ad("data/interim/unperturbed_qc.h5ad")

    obs_df: pd.DataFrame = cast(pd.DataFrame, ad.obs)
    candidate_cols = ["n_counts", "n_genes_by_counts", "pct_counts_mt"]
    cols: list[str] = [c for c in candidate_cols if c in obs_df.columns]
    obs_num: pd.DataFrame = obs_df[cols].apply(pd.to_numeric, errors="coerce").copy() # type: ignore

    qc_summary: pd.DataFrame = obs_num.describe()
    qc_summary.to_csv(out_dir / "qc_summary.csv")

    # ---- Plots ----
    if cols:
        sc.settings.figdir = str(out_dir)
        sc.pl.violin(ad, cols, show=False, save="_qc_violin.png")  # type: ignore
        sc.pl.highly_variable_genes(ad, show=False, save="_hvg.png")

    # ---- Manifest ----
    manifest: Dict[str, Any] = {
        "time": time.time(),
        "git": os.popen("git rev-parse --short HEAD").read().strip(),
        "inputs": [args.adata],
        "outputs": [
            str(out_dir / "qc_summary.csv"),
            "figures/qc_violin.png",
            "figures/hvg.png",
        ],
        "params": params,
    }
    (out_dir / "manifest_qc.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
