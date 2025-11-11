#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, cast

import pandas as pd
import scanpy as sc  # type: ignore
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

    print("\n".join(list(ad.obs)))
    print()
    print("\n".join(list(ad.var)))

    conver_col = []

    for col in ad.var.columns:
        # Try converting the column to numeric
        # 'errors="coerce"' will turn any non-numeric values into NaN (Not a Number)
        numeric_col = pd.to_numeric(ad.var[col], errors="coerce")

        # Check if the conversion was successful and if there were non-numeric values
        # If there were non-numeric values (resulting in NaNs), you might need to handle them
        if numeric_col.notna().all():
            # If all values are now numeric and the type changed, replace the original column
            ad.var[col] = ad.var[col].astype(float)
            ad.var[col] = ad.var[col].astype(int)
            conver_col.append(col)
            # print(f"Converted column '{col}' to numeric.")
        else:
            # Optionally, handle columns that couldn't be fully converted
            print(
                f"Column {col} remains as is (might contain non-numeric data or already numeric."
            )

    priors = ["mean", "std", "cv", "fano", "mitopercent", "UMI_count"]
    qc_cols = [x for x in priors if x in conver_col]
    for col in conver_col:
        print(f"Converted column '{col}' to numeric.")
        print(ad.var[col])
    for x in qc_cols:
        print("qc", x)

    # ---- QC metrics ----
    sc.pp.calculate_qc_metrics(
        ad,
        qc_vars=qc_cols,
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # Filters (explicit casts for Pylance)
    min_genes = int(params["min_genes_per_cell"])
    pct_mito_max = float(params["pct_mito_max"])
    ad = ad[ad.obs["n_genes_by_counts"] > min_genes, :]
    ad = ad[ad.obs["mitopercent"] < pct_mito_max, :]

    # Normalize + log + HVGs
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(
        ad, n_top_genes=int(params["hvg_n_top_genes"]), flavor="seurat_v3"
    )

    # Persist QC’d AnnData for downstream steps
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    ad.write_h5ad("data/interim/unperturbed_qc.h5ad")

    # ---- EDA summary with explicit pandas types ----
    obs_df: pd.DataFrame = cast(pd.DataFrame, ad.obs)
    cols: list[str] = ["n_counts", "n_genes_by_counts", "pct_counts_mt"]
    obs_num: pd.DataFrame = obs_df[cols].apply(pd.to_numeric, errors="coerce").copy() # type: ignore

    qc_summary: pd.DataFrame = obs_num.describe()
    qc_summary.to_csv(out_dir / "qc_summary.csv")

    # ---- Plots ----
    sc.pl.violin(ad, cols, show=False, save="_qc_violin.png") # type: ignore
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
