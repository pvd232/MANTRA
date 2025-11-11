#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Final, Any

import numpy as np
import pandas as pd
from pandas._typing import SeriesDType
from numpy.typing import NDArray


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Fit theta^(t) via weighted least squares."
    )
    ap.add_argument("--trait", required=True, help="Trait name (e.g., MCH)")
    ap.add_argument("--in", dest="inp", required=True, help="out/interim")
    ap.add_argument("--smr", required=True, help="data/smr directory")
    ap.add_argument("--out", required=True, help="out/interim")
    return ap


def wls_theta(
    D: NDArray[np.float64],
    s: NDArray[np.float64],
    se: NDArray[np.float64],
    ridge: float = 1e-6,
) -> NDArray[np.float64]:
    """
    Solve theta = argmin ||Sigma^{-1/2}(s - D theta)||^2 with small ridge.
    Shapes: D (genes×K), s (genes,), se (genes,).
    """
    w: NDArray[np.float64] = 1.0 / np.maximum(se, 1e-12) ** 2
    Si: NDArray[np.float64] = np.diag(w)
    A: NDArray[np.float64] = D.T @ Si @ D + ridge * np.eye(D.shape[1], dtype=np.float64)
    b: NDArray[np.float64] = D.T @ Si @ s
    theta: NDArray[np.float64] = np.linalg.solve(A, b).astype(np.float64, copy=False)
    return theta


def main() -> None:
    args = build_argparser().parse_args()

    w_path: Final[Path] = Path(args.inp) / "program_loadings_W.csv"
    W_df: pd.DataFrame = pd.read_csv(w_path, index_col=0) # type: ignore

    smr_path: Final[Path] = Path(args.smr) / f"{args.trait}_gene_effects.csv"
    smr_df: pd.DataFrame = pd.read_csv(smr_path)  # type: ignore # expects: gene, s, se

    D: NDArray[np.float64] = (
        W_df.reindex(smr_df["gene"]).fillna(0.0).to_numpy(dtype=np.float64) # type: ignore
    )  # genes×K
    s: NDArray[np.float64] = smr_df["s"].to_numpy(dtype=np.float64)  # genes
    se: NDArray[np.float64] = smr_df["se"].to_numpy(dtype=np.float64)  # genes

    theta: NDArray[np.float64] = wls_theta(D, s, se, ridge=1e-6)
    out_theta = Path(args.out) / f"theta_{args.trait}.csv"
    pd.Series(theta, index=W_df.columns, name="theta").to_csv(out_theta)

    manifest: dict[str, float | str | Any | list[str]] = {
        "time": time.time(),
        "git": os.popen("git rev-parse --short HEAD").read().strip(),
        "trait": args.trait,
        "inputs": ["program_loadings_W.csv", f"{args.trait}_gene_effects.csv"],
        "outputs": [str(out_theta)],
    }
    (Path(args.out) / "manifest_theta.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
