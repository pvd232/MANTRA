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
from numpy.typing import NDArray


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Map DeltaE to trait predictions.")
    ap.add_argument("--in", dest="inp", required=True, help="out/interim")
    ap.add_argument(
        "--gwps", required=True, help="dir containing beta.parquet and u_batch.parquet"
    )
    ap.add_argument("--out", required=True, help="out/interim")
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    # Load W and theta (MCH by default for interim)
    W_path: Final[Path] = Path(args.inp) / "program_loadings_W.csv"
    W: NDArray[np.float64] = pd.read_csv(W_path, index_col=0).to_numpy(dtype=np.float64) # type: ignore

    theta_path: Final[Path] = Path(args.inp) / "theta_MCH.csv"
    theta_df = pd.read_csv( # type: ignore
        theta_path, index_col=0
    )  # single column named "theta" (from 03)
    theta: NDArray[np.float64] = theta_df.iloc[:, 0].to_numpy(dtype=np.float64)  # (K,)

    # Load GWPS pieces
    B: NDArray[np.float64] = pd.read_parquet(Path(args.gwps) / "beta.parquet").to_numpy( # type: ignore
        dtype=np.float64
    )  # genes×R
    U: NDArray[np.float64] = pd.read_parquet( # type: ignore
        Path(args.gwps) / "u_batch.parquet"
    ).to_numpy(
        dtype=np.float64
    )  # R×N

    # Compute predictions
    DeltaE: NDArray[np.float64] = B @ U  # genes×N
    Delta_a: NDArray[np.float64] = W.T @ DeltaE  # K×N
    yhat: NDArray[np.float64] = theta @ Delta_a  # (N,)

    out_yhat = Path(args.out) / "yhat_baseline_MCH.csv"
    pd.Series(yhat, name="yhat").to_csv(out_yhat, index=False)

    manifest: dict[str,Any] = {
        "time": time.time(),
        "git": os.popen("git rev-parse --short HEAD").read().strip(),
        "inputs": [
            "beta.parquet",
            "u_batch.parquet",
            "program_loadings_W.csv",
            "theta_MCH.csv",
        ],
        "outputs": [str(out_yhat)],
    }
    (Path(args.out) / "manifest_pred.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
