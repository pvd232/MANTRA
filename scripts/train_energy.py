#!/usr/bin/env python3
# src/mantra/eggfm/train_energy.py
"""
Train an EGGFM energy model on a QC'd K562 AnnData, using either:

  - HVG expression (ad.X in the HVG-restricted gene space), or
  - a precomputed embedding stored in ad.obsm (e.g. "X_pca", "X_diffmap", "X_umap")

as the feature space for the EnergyMLP.

This module:
  - loads QC'd AnnData
  - optionally subsamples cells
  - restricts to HVGs (and top max_hvg by dispersion if configured)
  - selects the feature representation based on `latent_space`:
        * latent_space == "hvg" → use ad_prep.X
        * latent_space == "<key>" → use ad_prep.obsm["<key>"]
  - runs denoising score-matching (DSM) training for EnergyMLP
  - saves an energy checkpoint containing:
      * model state_dict
      * feature / gene names
      * mean / std normalizers in the chosen feature space
      * feature-space tag ("hvg", "X_pca", etc.)

Typical CLI wrapper usage (via scripts/train_energy.py):

  python scripts/train_energy.py \
      --params configs/params.yml \
      --ad data/interim/k562_gwps_unperturbed_qc.h5ad \
      --out out/models/eggfm \
      --space hvg

or, to train directly on an embedding:

  python scripts/train_energy.py \
      --params configs/params.yml \
      --ad data/interim/k562_gwps_unperturbed_hvg_embeddings.h5ad \
      --out out/models/eggfm \
      --space X_pca
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mantra.eggfm.run_energy import run_energy_training


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train EGGFM energy model on K562")
    p.add_argument(
        "--params",
        type=str,
        required=True,
        help="YAML params file (must contain eggfm_model and eggfm_train)",
    )
    p.add_argument(
        "--ad",
        type=str,
        required=True,
        help="Preprocessed K562 AnnData (e.g. data/interim/k562_replogle_prep.h5ad)",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for checkpoints (e.g. out/models/eggfm)",
    )
    p.add_argument(
        "--space",
        type=str,
        default="hvg",
        help="Representation to train on: 'hvg' or an .obsm key like 'X_pca'",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run_energy_training(
        params_path=Path(args.params),
        ad_path=Path(args.ad),
        out_dir=Path(args.out),
        space=args.space,
    )


if __name__ == "__main__":
    main()
