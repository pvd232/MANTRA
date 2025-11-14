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

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QC + EDA for unperturbed cells.")
    ap.add_argument("--params", required=True, help="configs/params.yml")
    ap.add_argument("--out", required=True, help="out/interim")
    ap.add_argument("--ad", required=True, help="path to unperturbed .h5ad")
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


def _auto_normalize_total(
    ad: sc.AnnData,
    target_sum: float = 1e4,
    frac_thresh: float = 0.2,
    min_genes_flag: int = 3,
) -> None:
    """
    Decide whether to use `exclude_highly_expressed` based on gene fractions,
    then call sc.pp.normalize_total in-place.

    frac_thresh: a gene is 'extreme' if in *some* cell it is >= this fraction
                 of that cell's total counts.
    min_genes_flag: if at least this many genes are 'extreme', we enable
                    exclude_highly_expressed.
    """
    X = ad.X

    # cell-wise totals on raw counts
    totals = np.asarray(X.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0  # avoid div-by-zero

    if sparse.issparse(X):
        X_csc = X.tocsc()
        n_genes = X_csc.shape[1]
        max_frac = np.zeros(n_genes, dtype=float)

        for j in range(n_genes):
            col = X_csc.getcol(j)
            if col.nnz == 0:
                continue
            rows = col.indices
            data = col.data
            fracs = data / totals[rows]
            if fracs.size:
                max_frac[j] = fracs.max()
    else:
        # dense: simple broadcasting
        fracs = X / totals[:, None]
        max_frac = fracs.max(axis=0)

    extreme_genes = np.where(max_frac >= frac_thresh)[0]
    n_extreme = len(extreme_genes)

    exclude = n_extreme >= min_genes_flag

    print(
        f"[normalize_total] target_sum={target_sum:.0f}, "
        f"exclude_highly_expressed={exclude} "
        f"({n_extreme} genes with max_frac >= {frac_thresh:.2f})"
    )

    sc.pp.normalize_total(
        ad,
        target_sum=target_sum,
        exclude_highly_expressed=exclude,
        # we can pass the same threshold we used for diagnostics
        max_fraction=frac_thresh,
    )


def is_integer_like_matrix(M) -> bool:
    data = M.data if sparse.issparse(M) else np.ravel(M)
    return (
        (data.size > 0)
        and np.isfinite(data).all()
        and np.allclose(data, np.round(data), atol=1e-8)
    )


def prep(ad: sc.AnnData, params: Dict[str, Any]):
    # Remove genes that are not statistically relevant
    n_cells = ad.n_obs

    # gene must appear in >=0.1% of cells
    min_cells = max(3, int(0.001 * n_cells))
    sc.pp.filter_genes(ad, min_cells=min_cells)

    # Remove empty droplets
    sc.pp.filter_cells(ad, min_genes=int(params["qc"]["min_genes"]))

    # Cells with high percent of mitochondrial DNA are dying or damaged
    ad = ad[ad.obs["mitopercent"] < float(params["qc"]["max_pct_mt"])].copy()

    # --- 2) Store raw counts before normalization ---
    # if "counts" not in ad.layers:
    #     ad.layers["counts"] = ad.X.copy()

    _auto_normalize_total(ad)
    sc.pp.log1p(ad)

    # Store log-norm count matrix
    # ad.layers["lognorm"] = ad.X.copy()

    flavor = "seurat_v3" if is_integer_like_matrix(ad.X) else "seurat"

    sc.pp.highly_variable_genes(
        ad, n_top_genes=int(params["hvg_n_top_genes"]), subset=True, flavor=flavor
    )
    # sc.pp.scale(ad, max_value=10)
    return ad


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


def report(ad: sc.AnnData, args: Dict[str], params: Dict[str]) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_files: List[Path] = []

    # QC summary CSV
    obs_cols = [
        c
        for c in ["n_counts", "n_genes_by_counts", "mitopercent", "pct_counts_mt"]
        if c in ad.obs
    ]
    if obs_cols:
        qc_summary = ad.obs[obs_cols].apply(pd.to_numeric, errors="coerce").describe()
        qc_csv = out_dir / "qc_summary.csv"
        qc_summary.to_csv(qc_csv)
        report_files.append(qc_csv)

    # Manifest JSON
    manifest = {
        "git": os.popen("git rev-parse --short HEAD").read().strip(),
        "input": os.path.abspath(args.ad),
        "params": {
            "min_genes": params["qc"]["min_genes"],
            "max_pct_mt": params["qc"]["max_pct_mt"],
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

    # Subsample for plotting
    nmax = int(args.plot_max_cells)
    if ad.n_obs > nmax:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(ad.n_obs, size=nmax, replace=False))
        ad_plot = ad[idx, :].copy()
        print(f"[plot] subsampled {nmax}/{ad.n_obs} for speed")
    else:
        ad_plot = ad

    # # PCA/Neighbors/UMAP if needed for nicer violins ordering later (optional)
    # if "X_pca" not in ad_plot.obsm:
    #     sc.pp.scale(ad_plot, max_value=10)
    #     sc.pp.pca(ad_plot)
    #     pca_png = out_dir / "K516_pca.png"
    #     report_files.append(pca_png)
    #     sc.pl.pca(ad_plot, svd_solver="arpack", save=pca_png)

    qc_png = out_dir / "qc_violin.png"
    try:
        sc.pl.violin(
            ad_plot,
            keys=["n_counts", "n_genes_by_counts", "mitopercent"],
            jitter=0.4,
            multi_panel=True,
            show=False,
            save=None,
        )
        plt.savefig(qc_png, bbox_inches="tight", dpi=160)
        plt.close()
        report_files.append(qc_png)
    except Exception as e:
        print(f"[plot] violin failed: {e}")

    # 2) HVG overview
    hvg_png = out_dir / "hvg.png"
    try:
        sc.pl.highly_variable_genes(ad_plot, show=False, save=None)
        plt.savefig(hvg_png, bbox_inches="tight", dpi=160)
        plt.close()
        report_files.append(hvg_png)
    except Exception as e:
        print(f"[plot] hvg plot failed: {e}")

    print(f"[report] wrote locally: {[p.name for p in report_files]}")

    # ---- Optional GCS upload (AFTER local writes) ----
    if args.report_to_gcs:
        results = _try_gsutil_cp(report_files, args.report_to_gcs)
        if results["uploaded"]:
            print("[report] uploaded to GCS:", ", ".join(results["uploaded"]))
        if results["failed"]:
            print(
                "[report] kept local copies for (upload failed):",
                ", ".join(results["failed"]),
            )
    else:
        print("[report] no --report-to-gcs provided; keeping local files only.")


def main() -> None:
    args = build_argparser().parse_args()
    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load AnnData
    ad = sc.read_h5ad(args.ad)
    print("ad shape", ad.shape)
    for col in ad.obs.columns:
        print("obs col:", col)
    for col in ad.var.columns:
        print("var col:", col)

    # QC processing
    qc_ad = prep(ad.copy(), params)

    # ---- persist ----
    qc_d_path = out_dir / "unperturbed_qc.h5ad"
    qc_ad.write_h5ad(qc_d_path)
    print("[main] after write_h5ad", flush=True)
    # ---- reporting (optional) ----
    # report(qc_ad)

    n_pcs = int(params["spec"]["n_pcs"])
    max_cells_dcol = int(params["spec"]["dcol_max_cells"])

    if qc_ad.n_obs > max_cells_dcol:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(qc_ad.n_obs, size=max_cells_dcol, replace=False))
        qc_dcol = qc_ad[idx, :].copy()
        print(
            f"[dcol_pca] subsampled {max_cells_dcol}/{qc_ad.n_obs} cells for DCOL-PCA"
        )

    else:
        qc_dcol = qc_ad
        print(f"[dcol_pca] using all {qc_ad.n_obs} cells for DCOL-PCA")

    X_sub = qc_dcol.X

    # Make a dense matrix for the *subset only*
    # and only if sparse
    print(
        "[dcol_pca] subset shape:", qc_dcol.shape, "sparse?", sparse.issparse(qc_dcol.X)
    )
    if sparse.issparse(qc_dcol.X):
        X_sub = X_sub.toarray()

    print("[main] before DCOL block", flush=True)
    K_sub = dcol_pca0(X_sub, nPC_max=n_pcs, Scale=True)
    vecs = K_sub["vecs"]  # shape n_genes x n_pcs

    # Project all cells using the same gene loadings
    X_full = qc_ad.X
    X_proj_full = X_full @ vecs
    print("[main] after DCOL block", flush=True)
    qc_ad.obsm["X_dcolpca"] = X_proj_full
    d_plot = plot_spectral(K_sub["vals"], out_dir, "dcol-pca")
    qc_ad.write_h5ad(d_plot)
    print("[main] b4 PCA", flush=True)
    # =========================
    # Optional: regular PCA for comparison
    # =========================
    # sc.tl.pca(qc_ad, n_comps=n_pcs, use_highly_variable=False)
    # sc.tl.pca(qc_ad, n_comps=n_pcs, use_highly_variable=False, zero_center=False)

    # pca_vals = qc_ad.uns["pca"]["variance"]
    # pca_plot = plot_spectral(pca_vals, out_dir, "reg-pca")

    # qc_pca_path = out_dir / "pca.h5ad"
    # qc_ad.write_h5ad(qc_pca_path)

    # Upload if requested
    # if args.report_to_gcs:
    #     _try_gsutil_cp([qc_d_path, qc_pca_path, d_plot, pca_plot], args.report_to_gcs)


if __name__ == "__main__":
    main()
