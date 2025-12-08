#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mantra.qc import run_qc


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QC + EDA for cells.")
    ap.add_argument("--params", required=True, help="configs/params.yml")
    ap.add_argument(
        "--out",
        required=True,
        help="Output QC AnnData .h5ad file (e.g. data/interim/k562_qc.h5ad)",
    )
    ap.add_argument("--ad", required=True, help="Path to input .h5ad")
    ap.add_argument(
        "--pet",
        action="store_true",
        help="If set, restrict to non-targeting control cells (gene == 'non-targeting')",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    run_qc(
        params_path=Path(args.params),
        ad_path=Path(args.ad),
        out_path=Path(args.out),
        pet=args.pet,
    )


if __name__ == "__main__":
    main()
