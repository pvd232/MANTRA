#!/usr/bin/env python3
# src/mantra/scripts/strip_uns_log1p.py
"""
Remove the /uns/log1p group from an .h5ad file to fix
AnnData IORegistryError issues caused by incompatible log1p metadata.

This script:
  - optionally copies the input .h5ad before editing
  - opens the target file with h5py
  - deletes /uns/log1p if present
  - leaves other contents untouched

Usage:

  python scripts/strip_uns_log1p.py \
      --h5ad-in data/raw/some_dataset.h5ad \
      --h5ad-out data/raw/some_dataset_nolog1p.h5ad
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py  # type: ignore


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Remove /uns/log1p from an .h5ad file to fix AnnData IORegistryError."
    )
    p.add_argument(
        "--h5ad-in",
        required=True,
        help="Input .h5ad file (will be copied if --h5ad-out is set).",
    )
    p.add_argument(
        "--h5ad-out",
        default=None,
        help="Optional output .h5ad file. "
             "If omitted, edits are done in-place on --h5ad-in.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    src = Path(args.h5ad_in)
    if not src.exists():
        raise SystemExit(f"Input file not found: {src}")

    # Decide where to write
    if args.h5ad_out is None:
        dst = src
        print(f"[info] editing in-place: {dst}")
    else:
        dst = Path(args.h5ad_out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[info] copying {src} -> {dst}")
        shutil.copy2(src, dst)

    # Open with h5py and delete /uns/log1p if present
    with h5py.File(dst, "r+") as f:
        if "uns" not in f:
            print("[info] no /uns group found; nothing to remove.")
            return
        uns_grp = f["uns"]
        if "log1p" in uns_grp:
            print("[info] found /uns/log1p; deleting...")
            del uns_grp["log1p"]
            print("[done] /uns/log1p removed.")
        else:
            print("[info] /uns/log1p not present; nothing to remove.")


if __name__ == "__main__":
    main()
