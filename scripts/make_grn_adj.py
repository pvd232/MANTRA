#!/usr/bin/env python3
# src/mantra/scripts/make_grn_adj.py
"""
Build a correlation-based gene–gene adjacency matrix A [G, G]
in the EGGFM HVG space, aligned to the energy checkpoint var_names.

This script:
  - loads a QC'd unperturbed K562 AnnData (same used for EGGFM)
  - loads an EGGFM checkpoint to get HVG gene ordering
  - subsets AnnData to that gene set and ordering
  - computes gene–gene Pearson correlations across cells
  - thresholds by |corr| >= corr_thresh
  - optionally keeps top-k neighbors per gene
  - row-normalizes to obtain a stochastic adjacency for message passing
  - saves A as a .npy file for use with GRNGNN (--adj flag)

Typical usage:

  python src/mantra/scripts/make_grn_adj.py \\
      --ad data/interim/k562_gwps_unperturbed_qc.h5ad \\
      --energy-ckpt out/models/eggfm/eggfm_energy_k562_hvg_hvg100.pt \\
      --out data/interim/A_k562_hvg100.npy \\
      --corr-thresh 0.2 \\
      --topk 10

"""

from __future__ import annotations

import argparse
from pathlib import Path

from mantra.grn.make_adj import make_grn_adj


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build a correlation-based gene–gene adjacency [G,G] "
            "aligned to an EGGFM HVG checkpoint."
        )
    )

    p.add_argument(
        "--ad",
        type=str,
        required=True,
        help="QC'd unperturbed AnnData (e.g. data/interim/k562_gwps_unperturbed_qc.h5ad)",
    )
    p.add_argument(
        "--energy-ckpt",
        type=str,
        required=True,
        help="EGGFM .pt checkpoint with 'var_names' or 'feature_names' for HVGs",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output .npy path for adjacency [G,G]",
    )
    p.add_argument(
        "--corr-thresh",
        type=float,
        default=0.2,
        help="Min absolute Pearson correlation to keep an edge (default: 0.2)",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Optional top-k neighbors per gene (per row); set <=0 to disable.",
    )

    return p


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    stats = make_grn_adj(
        ad_path=Path(args.ad),
        energy_ckpt_path=Path(args.energy_ckpt),
        out_path=Path(args.out),
        corr_thresh=args.corr_thresh,
        topk=args.topk if args.topk > 0 else None,
    )

    print(
        f"[adj][summary] G_eff={stats['G_eff']}, "
        f"corr_thresh={stats['corr_thresh']}, "
        f"topk={stats['topk']}, "
        f"n_edges={stats['n_edges']}, "
        f"out={stats['out_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
