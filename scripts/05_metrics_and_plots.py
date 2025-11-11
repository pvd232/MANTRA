#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    r2_score,
    roc_auc_score,
)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compute metrics and calibration.")
    ap.add_argument("--in", dest="inp", required=True, help="out/interim")
    ap.add_argument("--out", required=True, help="out/interim")
    return ap


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    args = build_argparser().parse_args()

    # yhat saved as a single-column CSV ("yhat")
    yhat_path: Final[Path] = Path(args.inp) / "yhat_baseline_MCH.csv"
    yhat_df = pd.read_csv(yhat_path) # type: ignore
    yhat: NDArray[np.float64] = yhat_df.iloc[:, 0].to_numpy(dtype=np.float64)

    # Truth labels: columns y_cont (float), y_sign (0/1)
    truth_df = pd.read_csv("data/labels/kd_trait_mch.csv") # type: ignore
    y: NDArray[np.float64] = truth_df["y_cont"].to_numpy(dtype=np.float64)
    ybin: NDArray[np.int_] = truth_df["y_sign"].to_numpy(dtype=np.int_)  # 0/1

    # Metrics
    r2 = float(r2_score(y, yhat))
    auroc = float(roc_auc_score(ybin, yhat))
    aupr = float(average_precision_score(ybin, yhat))

    prob: NDArray[np.float64] = sigmoid(yhat)
    frac, meanp = calibration_curve(ybin, prob, n_bins=10, strategy="uniform")
    brier = float(brier_score_loss(ybin, prob))

    # Save
    out_dir = Path(args.out)
    (out_dir / "metrics_baseline_MCH.csv").write_text(
        pd.DataFrame(
            {
                "metric": ["R2", "AUROC", "AUPRC", "Brier"],
                "value": [r2, auroc, aupr, brier],
            }
        ).to_csv(index=False)
    )
    np.savetxt(
        out_dir / "calibration_curve_MCH.tsv",
        np.c_[meanp, frac],
        fmt="%.4f",
        header="mean_pred\tfraction_of_positives",
    )


if __name__ == "__main__":
    main()
