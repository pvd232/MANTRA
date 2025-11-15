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


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QC + EDA for unperturbed cells.")
    ap.add_argument("--params", required=True, help="configs/params.yml")
    ap.add_argument("--out", required=True, help="out/interim")
    ap.add_argument("--adata", required=True, help="path to unperturbed .h5ad")
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


def _try_gsutil_cp(paths: List[Path], gs_prefix: str) -> Dict[str, List[str]]:
    """
    Try to 'gsutil cp' each file. Always keep local copies.
    Returns a dict with 'uploaded' and 'failed' lists of basenames.
    """
    results: dict[str, list[str]] = {"uploaded": [], "failed": []}
    gs_prefix = gs_prefix.rstrip("/")

    for p in paths:
        try:
            # Use -n so we don't overwrite; capture output for debugging
            proc = subprocess.run(
                ["gsutil", "-m", "cp", "-n", str(p), f"{gs_prefix}/"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode == 0:
                results["uploaded"].append(p.name)
            else:
                # Do not raise; we want to keep going and keep files local.
                results["failed"].append(p.name)
                print(f"[report] upload failed for {p.name}:\n{proc.stderr.strip()}")
        except FileNotFoundError:
            # gsutil not installed
            results["failed"].append(p.name)
            print("[report] 'gsutil' not found on PATH; keeping file locally:", p.name)
        except Exception as e:
            results["failed"].append(p.name)
            print(f"[report] unexpected error uploading {p.name}: {e}")
    return results


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

    # --- choose the counts matrix (prefer true counts) ---
    if "counts" in (ad.layers or {}):
        X_counts = ad.layers["counts"]
        counts_src = "layers['counts']"
    elif ad.raw is not None:
        X_counts = ad.raw.X
        counts_src = "raw.X"
    else:
        X_counts = ad.X
        counts_src = "X (assumed counts)"
    print(f"[QC] Using {counts_src} as counts source")

    # Drop zero-count cells
    totals = np.ravel(X_counts.sum(axis=1))
    keep = totals > 0
    if not keep.all():
        print(f"[QC] Dropping {(~keep).sum()} zero-count cells before normalize/log")
        ad = ad[keep, :].copy()
        X_counts = X_counts[keep, :]

    # Ensure basic QC columns from your schema
    # n_counts from totals / umi_count if present
    if "umi_count" in ad.obs.columns:
        ad.obs["n_counts"] = pd.to_numeric(ad.obs["umi_count"], errors="coerce")
    else:
        ad.obs["n_counts"] = totals

    # n_genes_by_counts if missing
    if "n_genes_by_counts" not in ad.obs.columns:
        if sparse.issparse(X_counts):
            ad.obs["n_genes_by_counts"] = (X_counts > 0).sum(axis=1).A1
        else:
            ad.obs["n_genes_by_counts"] = (np.asarray(X_counts) > 0).sum(axis=1)

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
    min_genes = int(params["qc"]["min_genes"])
    pct_mito_max = float(params["qc"]["pct_mito_max"])
    mask = (ad.obs["n_genes_by_counts"] > min_genes) & (
        ad.obs["mitopercent"] < pct_mito_max
    )
    print(f"[filter] keep {int(mask.sum())}/{ad.n_obs} cells after thresholds")
    ad = ad[mask, :].copy()
    X_counts = (
        X_counts[mask, :]
        if sparse.issparse(X_counts)
        else np.asarray(X_counts)[mask, :]
    )

    # ---- normalize/log on counts ----
    ad.X = X_counts
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

    flavor = "seurat_v3" if is_integer_like_matrix(X_counts) else "seurat"
    print(f"[HVG] Using flavor={flavor}")
    sc.pp.highly_variable_genes(
        ad, n_top_genes=int(params["hvg_n_top_genes"]), flavor=flavor
    )
    print("done")
    # ---- persist ----
    # Path("data/interim").mkdir(parents=True, exist_ok=True)
    # ad.write_h5ad("data/interim/unperturbed_qc.h5ad")

    # ---- reporting (optional) ----
    # report_files: List[Path] = []
    # if args.report:
    #     out_dir = Path(args.out)
    #     out_dir.mkdir(parents=True, exist_ok=True)

    #     # QC summary CSV
    #     obs_cols = [
    #         c
    #         for c in ["n_counts", "n_genes_by_counts", "mitopercent", "pct_counts_mt"]
    #         if c in ad.obs
    #     ]
    #     if obs_cols:
    #         qc_summary = (
    #             ad.obs[obs_cols].apply(pd.to_numeric, errors="coerce").describe()
    #         )
    #         qc_csv = out_dir / "qc_summary.csv"
    #         qc_summary.to_csv(qc_csv)
    #         report_files.append(qc_csv)

    # Manifest JSON
    # manifest = {
    #     "git": os.popen("git rev-parse --short HEAD").read().strip(),
    #     "input": os.path.abspath(args.adata),
    #     "params": {
    #         "min_genes": min_genes,
    #         "pct_mito_max": pct_mito_max,
    #         "hvg_n_top_genes": int(params["hvg_n_top_genes"]),
    #     },
    #     "n_cells": int(ad.n_obs),
    #     "n_genes": int(ad.n_vars),
    #     "obs_cols": list(ad.obs.columns)[:25],
    #     "var_cols": list(ad.var.columns)[:25],
    # }
    # man_json = out_dir / "manifest_qc.json"
    # # man_json.write_text(json.dumps(manifest, indent=2))
    # report_files.append(man_json)

    # # Subsample for plotting
    # nmax = int(args.plot_max_cells)
    # if ad.n_obs > nmax:
    #     rng = np.random.default_rng(0)
    #     idx = np.sort(rng.choice(ad.n_obs, size=nmax, replace=False))
    #     ad_plot = ad[idx, :].copy()
    #     print(f"[plot] subsampled {nmax}/{ad.n_obs} for speed")
    # else:
    #     ad_plot = ad

    # # PCA/Neighbors/UMAP if needed for nicer violins ordering later (optional)
    # if "X_pca" not in ad_plot.obsm:
    #     sc.pp.scale(ad_plot, max_value=10)
    #     sc.pl.pca(ad_plot, svd_solver="arpack", save=out_dir / "K516_pca.png")

    # qc_png = out_dir / "qc_violin.png"
    # try:
    #     sc.pl.violin(
    #         ad_plot,
    #         keys=["n_counts", "n_genes_by_counts", "mitopercent"],
    #         jitter=0.4,
    #         multi_panel=True,
    #         show=False,
    #         save=None,
    #     )
    #     plt.savefig(qc_png, bbox_inches="tight", dpi=160)
    #     plt.close()
    #     report_files.append(qc_png)
    # except Exception as e:
    #     print(f"[plot] violin failed: {e}")

    # 2) HVG overview
    # hvg_png = out_dir / "hvg.png"
    # try:
    #     sc.pl.highly_variable_genes(ad_plot, show=False, save=None)
    #     plt.savefig(hvg_png, bbox_inches="tight", dpi=160)
    #     plt.close()
    #     report_files.append(hvg_png)
    # except Exception as e:
    #     print(f"[plot] hvg plot failed: {e}")

    # print(f"[report] wrote locally: {[p.name for p in report_files]}")

    # ---- Optional GCS upload (AFTER local writes) ----
    # if args.report_to_gcs:
    #     results = _try_gsutil_cp(report_files, args.report_to_gcs)
    #     if results["uploaded"]:
    #         print("[report] uploaded to GCS:", ", ".join(results["uploaded"]))
    #     if results["failed"]:
    #         print(
    #             "[report] kept local copies for (upload failed):",
    #             ", ".join(results["failed"]),
    #         )
    # else:
    #     print("[report] no --report-to-gcs provided; keeping local files only.")


if __name__ == "__main__":
    main()
