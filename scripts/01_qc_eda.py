#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore
from scipy import sparse
import subprocess
import yaml


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QC + EDA for unperturbed cells.")
    ap.add_argument("--params", required=True, help="configs/params.yml")
    ap.add_argument("--out", required=True, help="out/interim")
    ap.add_argument("--adata", required=True, help="path to unperturbed .h5ad")
    # reporting (you referenced these—now they exist)
    ap.add_argument(
        "--report", action="store_true", help="emit qc_summary, plots, manifest"
    )
    ap.add_argument(
        "--report-to-gcs",
        metavar="GS_PREFIX",
        default=None,
        help="If set (e.g., gs://BUCKET/out/interim), upload report files there",
    )
    ap.add_argument(
        "--plot-max-cells",
        type=int,
        default=10000,
        help="Max cells to plot (subsample if larger)",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load AnnData
    ad = sc.read_h5ad(args.adata).copy()
    print("ad", ad.shape)

    # --- subset to UNPERTURBED cells (adjust column names as needed) ---
    obs_lower = ad.obs.columns.str.lower()
    if any(c in obs_lower for c in ["guide", "sgrna", "target", "is_perturbed"]):
        col = [
            c
            for c in ad.obs.columns
            if c.lower() in ["guide", "sgrna", "target", "is_perturbed"]
        ][0]
        v = ad.obs[col].astype(str).str.lower()
        unpert_mask = v.isin(
            ["", "none", "nan", "nt", "control", "non-targeting", "false"]
        )
        print(f"[subset] keeping {int(unpert_mask.sum())} unperturbed of {ad.n_obs}")
        ad = ad[unpert_mask, :].copy()

    # --- choose the counts matrix (required for Seurat v3) ---
    if "counts" in (ad.layers or {}):
        X_counts = ad.layers["counts"]
        counts_src = "layers['counts']"
    elif ad.raw is not None:
        X_counts = ad.raw.X
        counts_src = "raw.X"
    else:
        raise RuntimeError(
            "Raw counts not found. Use replogle_counts.h5ad (has integer-like counts)."
        )
    print(f"[QC] Using {counts_src} as counts source")

    # Drop zero-count cells (on counts source)
    totals = np.ravel(X_counts.sum(axis=1))
    keep = totals > 0
    if not keep.all():
        print(f"[QC] Dropping {(~keep).sum()} zero-count cells before normalize/log")
        ad = ad[keep, :].copy()
        X_counts = X_counts[keep, :]

    # ---- mito percent ----
    if "mitopercent" not in ad.obs:
        var = ad.var
        mt_mask = None
        if "chr" in var.columns:
            mt_mask = (
                var["chr"].astype(str).str.upper().isin(["MT", "M", "CHRMT", "MITO"])
            )
        if (
            mt_mask is None or not bool(np.any(mt_mask))
        ) and "gene_name" in var.columns:
            mt_mask = var["gene_name"].astype(str).str.upper().str.startswith("MT-")
        if mt_mask is None:
            mt_mask = pd.Series(False, index=var.index)

        sc.pp.calculate_qc_metrics(
            ad,
            qc_vars={"mito": mt_mask.values},
            percent_top=None,
            log1p=False,
            inplace=True,
        )
        ad.obs["mitopercent"] = (
            ad.obs["pct_counts_mito"]
            if "pct_counts_mito" in ad.obs
            else ad.obs.get("pct_counts_mt", 0)
        )
    else:
        ad.obs["mitopercent"] = pd.to_numeric(ad.obs["mitopercent"], errors="coerce")
        if ad.obs["mitopercent"].max() <= 1.0:
            ad.obs["mitopercent"] = 100.0 * ad.obs["mitopercent"]

    # ---- filters ----
    min_genes = int(params["min_genes_per_cell"])
    pct_mito_max = float(params["pct_mito_max"])
    mask = (ad.obs["n_genes_by_counts"] > min_genes) & (
        ad.obs["mitopercent"] < pct_mito_max
    )
    ad = ad[mask, :].copy()

    # ---- normalize/log on counts ----
    ad.X = X_counts  # ensure working on counts
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    # ---- HVGs ----
    def is_integer_like_matrix(M) -> bool:
        data = M.data if sparse.issparse(M) else np.ravel(M)
        return (
            (data.size > 0)
            and np.isfinite(data).all()
            and np.allclose(data, np.round(data), atol=1e-8)
        )

    # We already used log1p on counts; the integer-likeness should be checked pre-log.
    # Use the counts source check instead of ad.X (now log1p).
    flavor = "seurat_v3" if is_integer_like_matrix(X_counts) else "seurat"
    print(f"[HVG] Using flavor={flavor}")
    sc.pp.highly_variable_genes(
        ad, n_top_genes=int(params["hvg_n_top_genes"]), flavor=flavor
    )

    # ---- persist ----
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    ad.write_h5ad("data/interim/unperturbed_qc.h5ad")

    # ---- reporting (optional) ----
    def _maybe_upload(paths, gs_prefix: str | None):
        if not gs_prefix:
            return
        gs_prefix = gs_prefix.rstrip("/")
        for p in paths:
            try:
                subprocess.run(
                    ["gsutil", "-m", "cp", "-n", str(p), f"{gs_prefix}/"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception as e:
                print(f"[report] upload failed for {p}: {e}")

    report_files: list[Path] = []
    if args.report:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        # qc_summary.csv
        obs_cols = [
            c
            for c in ["n_counts", "n_genes_by_counts", "mitopercent", "pct_counts_mt"]
            if c in ad.obs
        ]
        if obs_cols:
            qc_summary = (
                ad.obs[obs_cols].apply(pd.to_numeric, errors="coerce").describe()
            )
            qc_csv = out_dir / "qc_summary.csv"
            qc_summary.to_csv(qc_csv)
            report_files.append(qc_csv)

        # manifest_qc.json
        manifest = {
            "git": os.popen("git rev-parse --short HEAD").read().strip(),
            "input": os.path.abspath(args.adata),
            "params": {
                "min_genes_per_cell": min_genes,
                "pct_mito_max": pct_mito_max,
                "hvg_n_top_genes": int(params["hvg_n_top_genes"]),
            },
            "n_cells": int(ad.n_obs),
            "n_genes": int(ad.n_vars),
            "obs_cols": list(ad.obs.columns)[:25],
            "var_cols": list(ad.var.columns)[:25],
        }
        man_json = out_dir / "manifest_qc.json"
        man_json.write_text(json.dumps(manifest, indent=2))
        report_files.append(man_json)

        # quick plots on a subsample
        nmax = int(args.plot_max_cells)
        if ad.n_obs > nmax:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(ad.n_obs, size=nmax, replace=False))
            ad_plot = ad[idx, :].copy()
            print(f"[plot] subsampled {nmax}/{ad.n_obs} for speed")
        else:
            ad_plot = ad

        sc.settings.figdir = str(out_dir)
        cols = [
            c
            for c in ["n_counts", "n_genes_by_counts", "mitopercent", "pct_counts_mt"]
            if c in ad_plot.obs
        ]
        if cols:
            sc.pl.violin(ad_plot, cols, show=False, save="_qc_violin.png")
            report_files.append(out_dir / "qc_violin.png")
        if "highly_variable" in ad.var.columns:
            sc.pl.highly_variable_genes(ad_plot, show=False, save="_hvg.png")
            report_files.append(out_dir / "hvg.png")

        if args.report_to_gcs:
            _maybe_upload(report_files, args.report_to_gcs)


if __name__ == "__main__":
    main()
