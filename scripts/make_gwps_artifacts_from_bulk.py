#!/usr/bin/env python3
from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import scanpy as sc  # works fine on CPU

def pick_control_mask(obs: pd.DataFrame) -> pd.Series:
    """Heuristics to flag control columns (non-targeting). Edit if your column names differ."""
    s = pd.Series(False, index=obs.index)
    candidates = ["NTC", "control", "non-target", "negative"]
    text = (obs.astype(str).agg(" ".join, axis=1)).str.lower()
    for tok in candidates:
        s |= text.str.contains(tok.lower())
    return s

def normalize_log1p_cpm(X: np.ndarray) -> np.ndarray:
    lib = X.sum(axis=0, keepdims=True).clip(min=1.0)
    X_cpm = X * (1e6 / lib)
    return np.log1p(X_cpm)

def main() -> None:
    ap = argparse.ArgumentParser(description="Build beta.parquet and u_batch.parquet from bulk H5AD.")
    ap.add_argument("--in_h5ad", required=True, help="K562_gwps_normalized_bulk_01.h5ad (or raw)")
    ap.add_argument("--out_dir", required=True, help="output directory, e.g., data/gwps")
    ap.add_argument("--use_raw_counts", action="store_true", help="treat matrix as raw; log1p-CPM normalize")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ad = sc.read_h5ad(args.in_h5ad)   # genes × perturbations

    # matrix as dense float
    X = ad.X.A if hasattr(ad.X, "A") else ad.X
    X = np.asarray(X, dtype=np.float64, order="F")

    if args.use_raw_counts:
        X = normalize_log1p_cpm(X)  # bring to comparable scale first

    obs = ad.obs.copy()
    if "regulator_id" in obs.columns:
        regs = obs["regulator_id"].astype("string")
    elif "target" in obs.columns:
        regs = obs["target"].astype("string")
    else:
        # last resort: use column names if present
        regs = pd.Series(ad.obs_names, index=ad.obs_names, dtype="string")

    ctrl_mask = pick_control_mask(obs)
    if not ctrl_mask.any():
        raise RuntimeError("Could not detect control columns (NTC/negative). Edit pick_control_mask().")

    # collapse replicates by regulator (mean over columns for that regulator)
    df = pd.DataFrame(X, index=ad.var_names, columns=regs.values)
    B_reg = df.groupby(axis=1, level=0).mean()  # genes × regulators

    # build a single control profile (mean over detected controls)
    ctrl_profile = df.loc[:, ctrl_mask.values].mean(axis=1)

    # define beta as (regulator profile - control) on the current scale
    B = (B_reg.subtract(ctrl_profile, axis=0)).astype(np.float32)

    # save beta
    B.to_parquet(out / "beta.parquet")

    # u_batch: simple one-hot over regulators (N = num regulators)
    regs_unique = pd.Index(B.columns, dtype="string")
    U = pd.DataFrame(np.eye(len(regs_unique), dtype=np.float32),
                     index=regs_unique, columns=[f"sample_{i}" for i in range(len(regs_unique))])
    U.to_parquet(out / "u_batch.parquet")

    # metadata helpers (optional, nice to have)
    pd.DataFrame({"gene_id": B.index}).to_csv(out / "genes.csv", index=False)
    pd.DataFrame({"regulator_id": regs_unique}).to_csv(out / "regulators.csv", index=False)

    print(f"Wrote {out/'beta.parquet'} and {out/'u_batch.parquet'}")

if __name__ == "__main__":
    main()
