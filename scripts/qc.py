#!/usr/bin/env python3
# src/mantra/scripts/qc.py
"""
Run QC + EDA for single-cell RNA-seq AnnData objects.

This script is a thin wrapper around `mantra.qc.run_qc` and:
  - loads a raw / full AnnData (optionally containing both perturbed + control cells)
  - optionally restricts to non-targeting **control** cells only
  - filters genes and cells based on basic thresholds
  - computes HVGs and log-normalized expression
  - writes a QC'd AnnData to disk

Usage:

  # QC on all cells (perturbed + controls)
  python scripts/qc.py \
      --params configs/params.yml \
      --ad data/raw/k562_gwps.h5ad \
      --out data/interim/k562_gwps_qc.h5ad

  # controls-only QC (non-targeting / unperturbed cells)
  python scripts/qc.py \
      --params configs/params.yml \
      --ad data/raw/k562_gwps.h5ad \
      --out data/interim/k562_gwps_unperturbed_qc.h5ad \
      --controls-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mantra.qc import run_qc


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QC + EDA for cells.")
    ap.add_argument("--params", required=True, help="Path to configs/params.yml")
    ap.add_argument(
        "--out",
        required=True,
        help="Output QC AnnData .h5ad file (e.g. data/interim/k562_qc.h5ad)",
    )
    ap.add_argument("--ad", required=True, help="Path to input .h5ad")
    ap.add_argument(
        "--controls-only",
        action="store_true",
        dest="controls_only",
        help="If set, restrict to non-targeting control cells (gene == 'non-targeting').",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    run_qc(
        params_path=Path(args.params),
        ad_path=Path(args.ad),
        out_path=Path(args.out),
        # `pet` in the underlying function really means "controls-only"
        pet=args.controls_only,
    )


if __name__ == "__main__":
    main()
