#!/usr/bin/env python3
# src/mantra/eggfm/train_grn.py
"""
Train the GRN GNN block on K562 with a pre-trained EGGFM energy prior.

This script:
  - parses GRN model / train / loss configs from YAML
  - loads a QC'd K562 AnnData (for x_ref computation and gene alignment)
  - loads train/val NPZs with regulator-level ΔE / ΔP / ΔY
  - builds adjacency A and program loadings W (if provided)
  - constructs an energy prior from an EGGFM checkpoint
  - trains the GRNGNN (+ optional TraitHead)
  - saves a GRN checkpoint with model weights and prior metadata

Usage:

  python scripts/train_grn.py \
      --params configs/params.yml \
      --out out/models/grn_hvg75 \
      --ad data/interim/k562_gwps_unperturbed_qc.h5ad \
      --train-npz data/interim/grn_k562_gwps_hvg75_npz/train.npz \
      --val-npz data/interim/grn_k562_gwps_hvg75_npz/val.npz \
      --energy-ckpt out/models/eggfm/eggfm_energy_k562_hvg_hvg75.pt \
      --cnmf-W out/programs/k562_cnmf_hvg75/k562_cnmf_hvg75_W_consensus.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mantra.grn.run_grn import run_grn_training


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train GRN GNN on K562 with pre-trained EGGFM energy prior"
    )

    p.add_argument(
        "--params",
        type=str,
        required=True,
        help="YAML params file (must contain grn_model, grn_train, grn_loss)",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for GRN checkpoints / logs",
    )
    p.add_argument(
        "--ad",
        type=str,
        required=True,
        help="Preprocessed K562 AnnData (same gene order as ΔE)",
    )
    p.add_argument(
        "--train-npz",
        type=str,
        required=True,
        help="NPZ with aggregated (reg_idx, deltaE, deltaP_obs, deltaY_obs, dose)",
    )
    p.add_argument(
        "--val-npz",
        type=str,
        default=None,
        help="Optional NPZ for validation set",
    )
    p.add_argument(
        "--adj",
        type=str,
        default=None,
        help="Optional .npy adjacency [G,G]. If not provided, uses identity.",
    )
    p.add_argument(
        "--cnmf-W",
        type=str,
        default=None,
        help="Optional .npy cNMF loadings W [G,K]. If missing, uses identity [G,G].",
    )
    p.add_argument(
        "--energy-ckpt",
        type=str,
        required=True,
        help="Path to pre-trained EGGFM energy checkpoint (.pt)",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    run_grn_training(
        params_path=Path(args.params),
        out_dir=Path(args.out),
        ad_path=Path(args.ad),
        train_npz_path=Path(args.train_npz),
        val_npz_path=Path(args.val_npz) if args.val_npz is not None else None,
        energy_ckpt_path=Path(args.energy_ckpt),
        adj_path=Path(args.adj) if args.adj is not None else None,
        cnmf_W_path=Path(args.cnmf_W) if args.cnmf_W is not None else None,
    )


if __name__ == "__main__":
    main()
