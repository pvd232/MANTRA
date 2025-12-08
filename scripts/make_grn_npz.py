#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mantra.grn.make_npz import make_grn_npz
'''
python scripts/make_grn_npz.py \
  --ad-raw data/raw/k562_gwps.h5ad \
  --energy-ckpt out/models/eggfm/eggfm_energy_k562_hvg_hvg75.pt \
  --out-dir data/interim/grn_k562_gwps_hvg75_npz \
  --reg-col gene \
  --control-value non-targeting \
  --max-pct-mt 0.2 \
  --min-umi 2000 \
  --min-cells-per-group 10 \
  --val-frac 0.2 \
  --seed 7
'''

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Streaming aggregation of K562 GWPS into train/val NPZs "
            "in the EGGFM HVG space, using the energy checkpoint var_names."
        )
    )

    p.add_argument(
        "--ad-raw",
        type=str,
        required=True,
        help="Big K562 GWPS AnnData (e.g. data/raw/k562_gwps.h5ad)",
    )
    p.add_argument(
        "--energy-ckpt",
        type=str,
        required=True,
        help="EGGFM .pt checkpoint with 'var_names' for HVGs",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output dir for train.npz / val.npz",
    )

    # obs columns
    p.add_argument(
        "--reg-col",
        type=str,
        default="gene",
        help="obs column with perturbed target gene / regulator",
    )
    p.add_argument(
        "--dose-col",
        type=str,
        default="gem_group",
        help="obs column to treat as 'dose' (kept for API; ignored in K562)",
    )
    p.add_argument(
        "--control-value",
        type=str,
        default="non-targeting",
        help="Value in reg-col that denotes control / non-targeting cells",
    )

    # QC thresholds
    p.add_argument(
        "--max-pct-mt",
        type=float,
        default=0.2,
        help="Max allowed mitopercent (fraction, e.g. 0.2 for 20%)",
    )
    p.add_argument(
        "--min-umi",
        type=float,
        default=2000.0,
        help="Min UMI_count per cell",
    )

    p.add_argument(
        "--min-cells-per-group",
        type=int,
        default=10,
        help="Min #cells per regulator to keep a sample",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of regulator-level samples to use for validation",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for train/val split",
    )

    p.add_argument(
        "--cnmf-W",
        type=str,
        default=None,
        help="Optional W.npy [G,K] for ΔP_obs; if missing, ΔP_obs = ΔE",
    )
    p.add_argument(
        "--traits-dim",
        type=int,
        default=3,
        help="Dimensionality of ΔY_obs stub (e.g. 3 for MCH, RDW, IRF)",
    )

    return p


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    stats = make_grn_npz(
        ad_raw_path=Path(args.ad_raw),
        energy_ckpt_path=Path(args.energy_ckpt),
        out_dir=Path(args.out_dir),
        reg_col=args.reg_col,
        dose_col=args.dose_col,
        control_value=args.control_value,
        max_pct_mt=args.max_pct_mt,
        min_umi=args.min_umi,
        min_cells_per_group=args.min_cells_per_group,
        val_frac=args.val_frac,
        seed=args.seed,
        cnmf_W_path=Path(args.cnmf_W) if args.cnmf_W is not None else None,
        traits_dim=args.traits_dim,
    )

    print(
        f"[summary] N={stats['N']} "
        f"(train={stats['N_train']}, val={stats['N_val']}), "
        f"G_eff={stats['G_eff']}, "
        f"train={stats['train_path']}, val={stats['val_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
