#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, sys
from typing import Optional, Dict, Any, Tuple, List

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import h5py


def human_bytes(n: float) -> str:
    if n is None:
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"


def choose_regulator_column(obs: pd.DataFrame) -> Optional[str]:
    # Prefer string/categorical columns with moderate-high cardinality (not per-cell unique)
    text_like = [
        c
        for c in obs.columns
        if pd.api.types.is_string_dtype(obs[c])
        or pd.api.types.is_categorical_dtype(obs[c])
    ]
    best, best_card = None, 0
    for c in text_like:
        card = int(obs[c].nunique(dropna=True))
        # typical regulator IDs: tens–thousands; exclude per-cell barcodes etc.
        if 5 <= card <= max(20000, obs.shape[0] // 5) and card > best_card:
            best, best_card = c, card
    return best


def detect_controls(obs: pd.DataFrame) -> int:
    text = (obs.astype(str).agg(" ".join, axis=1)).str.lower()
    ctrl_tokens = ("ntc", "control", "non-target", "negative", "scramble", "scrambled")
    mask = pd.Series(False, index=obs.index)
    for tok in ctrl_tokens:
        mask |= text.str.contains(tok, na=False)
    return int(mask.sum())


def suggested_mt_mask(var: pd.DataFrame) -> Optional[pd.Series]:
    # Works if you have gene symbols or chromosome-like columns
    for col in var.columns:
        print("col", col)
    if "MT" in var.columns and var["MT"].dtype == bool:
        return var["MT"]  # already present
    sym = None
    if "gene_symbols" in var.columns:
        sym = var["gene_symbols"].astype(str)
    else:
        # Use index as fallback
        try:
            sym = var.index.astype(str)
        except Exception:
            return None
    u = sym.str.upper()
    print("u", u)
    mt = u.str.startswith(("MT-", "MT.", "MT_")) | u.eq("MT")
    for col in ("chrom", "chromosome", "chromosome_name", "seqname"):
        if col in var.columns:
            mt = mt | var[col].astype(str).str.upper().isin(["MT", "M"])
    return mt


def describe_X_lowlevel(h5: h5py.File) -> Dict[str, Any]:
    """
    Inspect X without materializing it. Detect dense vs (csr/csc)-sparse,
    dtype, shape, nnz, and estimated dense/sparse sizes.
    """
    out: Dict[str, Any] = {
        "shape": None,
        "dtype": None,
        "is_sparse": None,
        "sparse_format": None,
        "nnz": None,
        "est_dense_bytes": None,
        "est_sparse_bytes": None,
    }
    if "X" not in h5:
        return out
    X = h5["X"]
    # AnnData encodes sparse arrays as a group with keys {data,indices,indptr,shape}
    if isinstance(X, h5py.Group) and set(X.keys()) >= {
        "data",
        "indices",
        "indptr",
        "shape",
    }:
        shp = tuple(int(i) for i in X["shape"][()])
        nnz = int(X["data"].shape[0])
        dt = X["data"].dtype.str
        out.update(
            {
                "shape": shp,
                "dtype": dt,
                "is_sparse": True,
                "sparse_format": "csr/csc (h5ad)",
                "nnz": nnz,
            }
        )
        # size estimates
        # data + indices + indptr (assume 4-byte int for indices/indptr unless h5 says otherwise)
        ind_dt = X["indices"].dtype.itemsize
        indptr_dt = X["indptr"].dtype.itemsize
        data_dt = X["data"].dtype.itemsize
        est_sparse = nnz * data_dt + nnz * ind_dt + (shp[0] + 1) * indptr_dt
        out["est_sparse_bytes"] = est_sparse
        # dense estimate if it were loaded as dense float32 (conservative)
        out["est_dense_bytes"] = shp[0] * shp[1] * 4
        return out
    else:
        # Dense dataset
        shp = X.shape
        dt = X.dtype.str
        out.update(
            {"shape": shp, "dtype": dt, "is_sparse": False, "sparse_format": None}
        )
        # No nnz cheaply; estimate dense bytes directly
        out["est_dense_bytes"] = np.prod(shp) * np.dtype(X.dtype).itemsize
        return out


def summarize_anndata(
    path: str,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    # Open file low-level
    with h5py.File(path, "r") as h5:
        x_desc = describe_X_lowlevel(h5)
        obsm_keys = list(h5.get("obsm", {}).keys()) if "obsm" in h5 else []
        varm_keys = list(h5.get("varm", {}).keys()) if "varm" in h5 else []
        layers = list(h5.get("layers", {}).keys()) if "layers" in h5 else []
    # Open anndata in backed mode to inspect obs/var (cheap)
    ad_b = sc.read_h5ad(path, backed="r")
    # Note: .obs/.var in backed mode are pandas dataframes (read fully but small)
    obs = ad_b.obs.copy()
    var = ad_b.var.copy()
    shape = (ad_b.n_obs, ad_b.n_vars)
    ad_b.file.close()  # close backed file handle promptly

    # Compute suggestions
    reg_col = choose_regulator_column(obs)
    n_ctrl = detect_controls(obs)
    mt_mask = suggested_mt_mask(var)
    mt_present = mt_mask is not None
    n_mt = int(mt_mask.sum()) if mt_present else None

    summary: Dict[str, Any] = {
        "path": path,
        "shape": shape,
        "X": x_desc,
        "n_obs": int(shape[0]),
        "n_vars": int(shape[1]),
        "obs_columns": list(map(str, obs.columns.tolist())),
        "var_columns": list(map(str, var.columns.tolist())),
        "obsm_keys": obsm_keys,
        "varm_keys": varm_keys,
        "layers": layers,
        "head_obs": obs.head(3).to_dict(orient="list"),
        "head_var": var.head(3).to_dict(orient="list"),
        "regulator_column_suggestion": reg_col,
        "reg_col_cardinality": (int(obs[reg_col].nunique()) if reg_col else None),
        "n_controls_detected": n_ctrl,
        "has_var_MT": bool(mt_present),
        "n_var_MT_true": n_mt,
        "est_dense_size": human_bytes(x_desc.get("est_dense_bytes")),
        "est_sparse_size": human_bytes(x_desc.get("est_sparse_bytes")),
    }
    return summary, obs, var


def write_qc_ready_with_mt(src: str, dst: str) -> Dict[str, Any]:
    adata = sc.read_h5ad(src)  # this will load X; only use on manageable files
    mt = suggested_mt_mask(adata.var)
    if mt is None:
        raise RuntimeError(
            "Could not infer var['MT']; no gene symbols / chromosome-like fields found."
        )
    adata.var["MT"] = mt.values
    # ensure CSR to reduce memory overhead in downstream scanpy (if sparse available)
    if hasattr(adata.X, "tocsr"):
        adata.X = adata.X.tocsr()
    adata.write(dst)
    return {"wrote_qc_ready": dst, "var_MT_true": int(adata.var["MT"].sum())}


def write_downsample(
    src: str, dst: str, n_cells: int, seed: int = 123
) -> Dict[str, Any]:
    # streaming-downsample: use backed read to choose indices, then load subset
    ad_b = sc.read_h5ad(src, backed="r")
    n = ad_b.n_obs
    n_take = min(n_cells, n)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=n_take, replace=False))
    ad_b.file.close()
    # Load subset (materialize once)
    ad = sc.read_h5ad(src)[idx, :]
    if hasattr(ad.X, "tocsr"):
        ad.X = ad.X.tocsr()
    ad.write(dst)
    return {"wrote_sample": dst, "n_cells": int(n_take)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Schema & size inspection for .h5ad (no RAM blowups)."
    )
    ap.add_argument("--h5ad", required=True, help="Input .h5ad")
    ap.add_argument(
        "--out", default="h5ad_schema_summary.json", help="Write JSON summary"
    )
    ap.add_argument(
        "--emit-qc-ready",
        metavar="OUT_H5AD",
        help="Write copy with var['MT'] added (loads X; use for manageable files)",
    )
    ap.add_argument(
        "--downsample",
        type=int,
        default=0,
        help="If >0, write a downsampled copy with N cells (safe for huge files)",
    )
    ap.add_argument(
        "--downsample-out",
        default="downsampled.h5ad",
        help="Output path for --downsample",
    )
    args = ap.parse_args()

    summary, obs, var = summarize_anndata(args.h5ad)

    # Actionable hints for loader/QC:
    hints: List[str] = []
    X = summary["X"]
    if (
        X["is_sparse"] is False
        and X["est_dense_bytes"]
        and X["est_dense_bytes"] > 16 * (1024**3)
    ):
        hints.append(
            "X appears dense and very large; prefer a smaller file or convert to sparse before QC."
        )
    if not summary["has_var_MT"]:
        hints.append(
            "var['MT'] absent; add it (use --emit-qc-ready) so scanpy QC metrics work."
        )
    if summary["regulator_column_suggestion"] is None:
        hints.append(
            "No obvious regulator column; inspect obs columns with medium cardinality and set explicitly."
        )
    if "n_genes_by_counts" not in obs.columns or "total_counts" not in obs.columns:
        hints.append(
            "Typical QC columns missing; 01_qc_eda.py should compute them (OK)."
        )
    summary["hints"] = hints

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Optional write-outs
    if args.emit_qc_ready:
        info = write_qc_ready_with_mt(args.h5ad, args.emit_qc_ready)
        print(json.dumps(info, indent=2))
    if args.downsample and args.downsample > 0:
        info = write_downsample(args.h5ad, args.downsample_out, args.downsample)
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
