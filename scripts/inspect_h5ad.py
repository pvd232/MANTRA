#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
import scanpy as sc
import pandas as pd
from typing import Optional


def choose_regulator_column(obs: pd.DataFrame) -> Optional[str]:
    # Prefer columns with many unique strings (likely regulator IDs)
    text_cols = [
        c
        for c in obs.columns
        if pd.api.types.is_string_dtype(obs[c]) or obs[c].dtype == "category"
    ]
    best, best_card = None, 0
    for c in text_cols:
        card = obs[c].nunique(dropna=True)
        # de-prioritize columns that look like guide IDs (thousands) vs regulator IDs (tens–hundreds)
        if 5 <= card <= 20000 and card > best_card:
            best, best_card = c, card
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out", default="h5ad_schema_summary.json")
    args = ap.parse_args()

    ad = sc.read_h5ad(args.h5ad)
    obs = ad.obs.copy()

    summary = {
        "shape": ad.shape,
        "obs_columns": list(obs.columns),
        "head_obs": obs.head(3).to_dict(orient="list"),
    }

    # Heuristic control detector preview
    text = (obs.astype(str).agg(" ".join, axis=1)).str.lower()
    ctrl_tokens = ["ntc", "control", "non-target", "negative"]
    ctrl_mask = pd.Series(False, index=obs.index)
    for tok in ctrl_tokens:
        ctrl_mask |= text.str.contains(tok)
    summary["n_controls_detected"] = int(ctrl_mask.sum())

    # Suggest a regulator column
    reg_col = choose_regulator_column(obs)
    summary["suggested_regulator_column"] = reg_col
    if reg_col:
        summary["reg_col_cardinality"] = int(obs[reg_col].nunique())

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
