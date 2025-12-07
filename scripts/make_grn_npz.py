#!/usr/bin/env python
# scripts/make_grn_npz.py

'''
python scripts/make_grn_npz.py \
  --ad-pert data/interim/K562_gwps_pert_qc_hvg100.h5ad \
  --out-dir data/interim/grn_hvg100/ \
  --cnmf-W data/interim/cnmf_W_hvg100.npy \
  --reg-col "target" \
  --dose-col "dose_bin" \
  --control-flag "is_control" \
  --val-frac 0.2 \
  --seed 7
'''

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import scanpy as sc
from scipy import sparse as sp_sparse


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate K562 GWPS into train/val NPZ for GRN GNN"
    )
    p.add_argument(
        "--ad-pert",
        type=str,
        required=True,
        help="Perturbed AnnData (cells × HVGs, QC’d; gene order = EGGFM/--ad)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for NPZs (train.npz, val.npz)",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of examples to use for validation (default: 0.2)",
    )
    p.add_argument(
        "--cnmf-W",
        type=str,
        default=None,
        help="Optional W.npy [G,K] for ΔP; if missing, ΔP = ΔE",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for train/val split",
    )

    # These let you adapt to your actual obs schema
    p.add_argument(
        "--reg-col",
        type=str,
        default="target",
        help="obs column with perturbed regulator identifier",
    )
    p.add_argument(
        "--dose-col",
        type=str,
        default="dose_bin",
        help="obs column with dose (discrete or continuous)",
    )
    p.add_argument(
        "--control-flag",
        type=str,
        default="is_control",
        help="obs boolean column indicating control cells (e.g. non-targeting guides)",
    )
    p.add_argument(
        "--min-cells",
        type=int,
        default=10,
        help="Minimum #cells per (reg,dose) to keep an example",
    )

    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.ad_pert}")
    ad = sc.read_h5ad(args.ad_pert)

    # X in HVG space; ensure dense float32
    X = ad.X
    if sp_sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)  # [cells, G]
    n_cells, G = X.shape
    print(f"[info] X shape: {X.shape} (cells × genes)")

    reg = ad.obs[args.reg_col].to_numpy()
    dose = ad.obs[args.dose_col].to_numpy()
    is_ctrl = ad.obs[args.control_flag].to_numpy().astype(bool)

    # Unique perturbed regulators (ignore controls)
    regs = np.unique(reg[~is_ctrl])
    print(f"[info] {len(regs)} perturbed regulators (non-control)")

    # Map regulator id -> index
    reg_to_idx = {r: i for i, r in enumerate(regs)}

    # Control baselines per dose-bin
    ctrl_means = {}
    dose_vals = np.unique(dose)
    print(f"[info] doses: {dose_vals}")

    for d in dose_vals:
        mask = is_ctrl & (dose == d)
        if mask.sum() == 0:
            # Fallback: all controls regardless of dose
            mask = is_ctrl
        ctrl_means[d] = X[mask].mean(axis=0, keepdims=True)  # [1, G]

    deltaE_list = []
    reg_idx_list = []
    dose_list = []

    for r in regs:
        for d in dose_vals:
            mask = (~is_ctrl) & (reg == r) & (dose == d)
            n_rd = int(mask.sum())
            if n_rd < args.min_cells:
                continue
            x_rd = X[mask].mean(axis=0, keepdims=True)      # [1, G]
            deltaE = x_rd - ctrl_means[d]                   # [1, G]

            deltaE_list.append(deltaE)
            reg_idx_list.append(reg_to_idx[r])
            dose_list.append(d)

    if len(deltaE_list) == 0:
        raise RuntimeError("No (reg, dose) combos with enough cells; check obs cols.")

    deltaE = np.vstack(deltaE_list).astype(np.float32)   # [N, G]
    reg_idx = np.array(reg_idx_list, dtype=np.int64)     # [N]
    dose_arr = np.array(dose_list, dtype=np.float32)     # [N]
    N = deltaE.shape[0]

    print(f"[agg] built {N} examples; G={G}")

    # ΔP: either use W or just identity
    if args.cnmf_W is not None:
        print(f"[load] cNMF W from {args.cnmf_W}")
        W = np.load(args.cnmf_W)                         # [G, K]
        if W.shape[0] != G:
            raise ValueError(
                f"W.shape[0] = {W.shape[0]} != G = {G}; "
                "gene order / HVG selection must match AnnData."
            )
        deltaP = deltaE @ W                              # [N, K]
    else:
        print("[info] No W provided; using ΔP = ΔE (programs = genes).")
        deltaP = deltaE.copy()                           # [N, G]

    # ΔY: stub as zeros for now (e.g. T=3 for MCH, RDW, IRF)
    T = 3
    deltaY = np.zeros((N, T), dtype=np.float32)

    # Train/val split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    N_val = int(args.val_frac * N)
    val_idx = perm[:N_val]
    train_idx = perm[N_val:]

    def save_npz(fname: Path, idx: np.ndarray) -> None:
        np.savez_compressed(
            fname,
            reg_idx=reg_idx[idx],
            deltaE=deltaE[idx],
            deltaP_obs=deltaP[idx],
            deltaY_obs=deltaY[idx],
            dose=dose_arr[idx],
        )
        print(f"[save] {fname} (N={len(idx)})")

    save_npz(out_dir / "train.npz", train_idx)
    save_npz(out_dir / "val.npz", val_idx)


if __name__ == "__main__":
    main()
