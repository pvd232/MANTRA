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
    keep = totals > 0
    if not keep.all():
        print(f"[QC] Dropping {(~keep).sum()} zero-count cells before normalize/log")
        ad = ad[keep, :].copy()
        X_counts = X_counts[keep, :]

    # --- make ad.X safe to operate on (CSR + no NaNs)
    ad.X = ad.X.tocsr() if sparse.issparse(ad.X) else ad.X
    if sparse.issparse(ad.X):
        d = ad.X.data
        if np.isnan(d).any():
            print("[QC] Found NaNs in X → setting NaNs to 0")
            d[np.isnan(d)] = 0.0
    else:
        if np.isnan(ad.X).any():
            print("[QC] Found NaNs in dense X → setting NaNs to 0")
            ad.X = np.nan_to_num(ad.X, nan=0.0)

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

    # --- decide whether we can/should normalize+log
    has_log1p_flag = bool(ad.uns.get("log1p"))
    xmin = (
        ad.X.data.min()
        if sparse.issparse(ad.X) and ad.X.data.size
        else (float(np.min(ad.X)) if (not sparse.issparse(ad.X) and ad.X.size) else 0.0)
    )

    do_norm_log = (
        (not has_log1p_flag) and (xmin >= 0.0) and (counts_src != "X (fallback)")
    )
    if do_norm_log:
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    else:
        reason = []
        if has_log1p_flag:
            reason.append("log1p flag present")
        if xmin < 0.0:
            reason.append(f"min(X)={xmin} < 0")
        if counts_src == "X (fallback)":
            reason.append("no raw counts; X looks normalized")
        print(f"[QC] Skipping normalize/log ({'; '.join(reason) or 'policy'})")

    # --- HVG flavor: only use v3 if counts are integer-like AND not fallback
    def is_integer_like_matrix(M):
        data = M.data if sparse.issparse(M) else np.ravel(M)
        return (
            data.size > 0
            and np.all(np.isfinite(data))
            and np.allclose(data, np.round(data), atol=1e-8)
        )

    print("-1")

    use_v3 = (counts_src != "X (fallback)") and is_integer_like_matrix(X_counts)

    flavor = "seurat_v3" if use_v3 else "seurat"

    print(f"[HVG] Using flavor={flavor}")

    sc.pp.highly_variable_genes(
        ad, n_top_genes=int(params["hvg_n_top_genes"]), flavor=flavor
    )

    # Persist QC’d AnnData for downstream steps
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    print("-2")

    ad.write_h5ad("data/interim/unperturbed_qc.h5ad")

    print("1")
    obs_df: pd.DataFrame = cast(pd.DataFrame, ad.obs)
    candidate_cols = ["n_counts", "n_genes_by_counts", "pct_counts_mt"]
    cols: list[str] = [c for c in candidate_cols if c in obs_df.columns]
    print("2")

    obs_num: pd.DataFrame = obs_df[cols].apply(pd.to_numeric, errors="coerce").copy() # type: ignore
    print("3")

    qc_summary: pd.DataFrame = obs_num.describe()
    qc_summary.to_csv(out_dir / "qc_summary.csv")
    print("4")

    # ---- Plots ----
    if cols:
        sc.settings.figdir = str(out_dir)
        print("5")

        sc.pl.violin(ad, cols, show=False, save="_qc_violin.png")  # type: ignore
        sc.pl.highly_variable_genes(ad, show=False, save="_hvg.png")

    print("6")

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
