#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import json
import yaml
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse


# ----------------------------
# CLI
# ----------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QC + EDA for unperturbed cells.")
    ap.add_argument("--params", required=True, help="configs/params.yml")
    ap.add_argument("--out", required=True, help="out/interim")
    ap.add_argument("--adata", required=True, help="path to unperturbed .h5ad")
    return ap


# ----------------------------
# Utilities
# ----------------------------
def _pick_qc_matrix(adata: AnnData):
    """
    Prefer count-like matrices for QC in this order:
      1) layers['counts']
      2) raw.X
      3) X
    Returns (matrix, source_tag).
    """
    if "counts" in (adata.layers or {}):
        return adata.layers["counts"], "layers['counts']"
    if adata.raw is not None:
        return adata.raw.X, "raw.X"
    return adata.X, "X"


def _mt_mask_from_var(var: pd.DataFrame) -> Optional[pd.Series]:
    """
    Robust mitochondrial gene detection using the *actual* schema you printed:
      - Prefer 'chr' if present (normalize to {M, MT, ...})
      - Else fallback to 'gene_name' using human/mouse conventions (MT-..., mt-...)
      - Else return None (skip MT gracefully)
    """
    if "chr" in var.columns:
        chr_norm = (
            var["chr"]
            .astype(str)
            .str.replace(r"^chr", "", case=False, regex=True)
            .str.strip()
            .str.upper()
        )
        mt = chr_norm.isin({"M", "MT", "MITO", "MITOCHONDRIAL"})
        # permissive safeguard
        mt |= chr_norm.str.contains(r"\bMT\b", regex=True)
        return mt.fillna(False)

    if "gene_name" in var.columns:
        syms = var["gene_name"].astype(str)
        return (syms.str.startswith("MT-") | syms.str.startswith("mt-")).fillna(False)

    return None


def _dense_sum_axis1(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.ravel(X.sum(axis=1))
    return np.ravel(X.sum(axis=1))


def _detected_genes_axis1(X) -> np.ndarray:
    if sparse.issparse(X):
        Xcsr = X.tocsr()
        return np.diff(Xcsr.indptr)
    return np.ravel((X > 0).sum(axis=1))


def _percent(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 * a / np.clip(b, 1e-12, None)
        out[np.isnan(out)] = 0.0
    return out


# ----------------------------
# QC core
# ----------------------------
def add_qc_metrics(adata: AnnData) -> Dict[str, Any]:
    """
    Adds:
      - obs['n_counts'], obs['n_genes_by_counts']
      - obs['pct_counts_mt'] if possible
    Returns a dict with provenance info for logging.
    """
    X, src = _pick_qc_matrix(adata)

    if "n_counts" not in adata.obs:
        adata.obs["n_counts"] = _dense_sum_axis1(X)

    if "n_genes_by_counts" not in adata.obs:
        adata.obs["n_genes_by_counts"] = _detected_genes_axis1(X)

    mt_mask = _mt_mask_from_var(adata.var)
    mt_status = "computed"
    if mt_mask is None or not bool(mt_mask.any()):
        mt_status = "skipped_no_mask"
    else:
        mt_mask = mt_mask.to_numpy(dtype=bool, na_value=False)
        if sparse.issparse(X):
            Xc = X.tocsc()
            mt_counts = np.ravel(Xc[:, mt_mask].sum(axis=1))
            totals = np.ravel(Xc.sum(axis=1))
        else:
            mt_counts = np.ravel(X[:, mt_mask].sum(axis=1))
            totals = np.ravel(X.sum(axis=1))
        adata.obs["pct_counts_mt"] = _percent(mt_counts, totals)

    return {
        "qc_matrix_source": src,
        "mt_detection": mt_status,
        "var_columns": list(map(str, adata.var.columns)),
    }


def apply_qc_filters(
    adata: AnnData,
    qc_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply simple threshold-based filters using config keys if present:
      qc:
        min_counts: int
        max_counts: int
        min_genes: int
        max_pct_mt: float
    Returns summary dict and writes a boolean 'pass_qc' column.
    """
    obs = adata.obs
    keep = pd.Series(True, index=obs.index)

    min_counts = qc_cfg.get("min_counts", None)
    if min_counts is not None and "n_counts" in obs:
        keep &= obs["n_counts"] >= int(min_counts)

    max_counts = qc_cfg.get("max_counts", None)
    if max_counts is not None and "n_counts" in obs:
        keep &= obs["n_counts"] <= int(max_counts)

    min_genes = qc_cfg.get("min_genes", None)
    if min_genes is not None and "n_genes_by_counts" in obs:
        keep &= obs["n_genes_by_counts"] >= int(min_genes)

    max_pct_mt = qc_cfg.get("max_pct_mt", None)
    if max_pct_mt is not None and "pct_counts_mt" in obs:
        keep &= obs["pct_counts_mt"] <= float(max_pct_mt)

    adata.obs["pass_qc"] = keep.values

    return {
        "n_cells_total": int(adata.n_obs),
        "n_cells_keep": int(keep.sum()),
        "n_cells_drop": int((~keep).sum()),
        "thresholds": {
            "min_counts": min_counts,
            "max_counts": max_counts,
            "min_genes": min_genes,
            "max_pct_mt": max_pct_mt,
        },
    }


# ----------------------------
# EDA plots
# ----------------------------
def make_qc_plots(adata: AnnData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Violin of standard QC metrics
    qc_keys = [
        k for k in ["n_counts", "n_genes_by_counts", "pct_counts_mt"] if k in adata.obs
    ]
    if qc_keys:
        try:
            sc.pl.violin(
                adata,
                qc_keys,
                jitter=0.4,
                multi_panel=True,
                show=False,
                save=None,
            )
            figpath = out_dir / "qc_violin.png"
            sc.pl.savefig(figpath.as_posix())
        except Exception:
            pass

    # Scatter: counts vs genes
    if {"n_counts", "n_genes_by_counts"} <= set(adata.obs.columns):
        try:
            ax = sc.pl.scatter(
                adata,
                x="n_counts",
                y="n_genes_by_counts",
                color="pct_counts_mt" if "pct_counts_mt" in adata.obs else None,
                show=False,
                return_fig=False,
                save=None,
            )
            figpath = out_dir / "qc_scatter_counts_vs_genes.png"
            sc.pl.savefig(figpath.as_posix())
        except Exception:
            pass

    # If you have embeddings already, add a quick UMAP colored by QC
    if "X_umap" in adata.obsm and qc_keys:
        for k in qc_keys:
            try:
                sc.pl.umap(adata, color=k, show=False, save=None)
                figpath = out_dir / f"qc_umap_{k}.png"
                sc.pl.savefig(figpath.as_posix())
            except Exception:
                pass


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = build_argparser().parse_args()
    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read AnnData in-memory for QC ops
    ad: AnnData = sc.read_h5ad(args.adata, backed=None)

    # Compute QC metrics robustly
    qc_prov = add_qc_metrics(ad)

    # Summaries pre-filter
    pre = {
        "n_cells": int(ad.n_obs),
        "n_genes": int(ad.n_vars),
        "qc_matrix_source": qc_prov["qc_matrix_source"],
        "mt_detection": qc_prov["mt_detection"],
    }

    # Apply filters if provided under params['qc']
    qc_cfg = dict(params.get("qc", {}))
    filt_summary = apply_qc_filters(ad, qc_cfg)

    # Save basic summaries
    (out_dir / "qc").mkdir(exist_ok=True, parents=True)
    with (out_dir / "qc" / "qc_summary.json").open("w") as f:
        json.dump(
            {
                "pre": pre,
                "filters": filt_summary,
                "var_columns": qc_prov["var_columns"],
            },
            f,
            indent=2,
        )

    # Write per-cell QC TSV
    qc_cols = [
        c
        for c in ["n_counts", "n_genes_by_counts", "pct_counts_mt", "pass_qc"]
        if c in ad.obs
    ]
    ad.obs.loc[:, qc_cols].to_csv(out_dir / "qc" / "per_cell_qc.tsv", sep="\t")

    # Make simple EDA plots
    make_qc_plots(ad, out_dir / "figs")

    # Save full (unfiltered) and filtered objects for downstream steps
    ad.write(out_dir / "adata_qc_unfiltered.h5ad", compression="gzip")
    if "pass_qc" in ad.obs and ad.obs["pass_qc"].any():
        ad_filt = ad[ad.obs["pass_qc"].values].copy()
    else:
        # If no filters defined or all passed, keep as-is
        ad_filt = ad.copy()
    ad_filt.write(out_dir / "adata_qc_filtered.h5ad", compression="gzip")

    # Console breadcrumbing
    print(
        json.dumps(
            {
                "loaded": str(args.adata),
                "out": str(out_dir),
                "qc_matrix_source": qc_prov["qc_matrix_source"],
                "mt_detection": qc_prov["mt_detection"],
                "cells_total": pre["n_cells"],
                "cells_keep": filt_summary["n_cells_keep"],
                "cells_drop": filt_summary["n_cells_drop"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
