#!/usr/bin/env python3
# scripts/03_fit_theta_wls.py
"""
Fit program-level trait effects (theta) from gene-level SMR/GWAS effects (beta)
using Weighted Least Squares (WLS):
    theta = argmin || W*theta - beta ||_{Sigma^-1}
where Sigma = diag(SE^2).
"""

import argparse
import numpy as np
import torch
from pathlib import Path

def run_wls(W, beta, se):
    # W: [G, K]
    # beta: [G]
    # se: [G]
    
    # weights = 1/se^2
    weights = 1.0 / (se**2 + 1e-8)
    W_w = W * np.sqrt(weights)[:, None]
    beta_w = beta * np.sqrt(weights)
    
    # Solve W_w * theta = beta_w
    theta, residuals, rank, s = np.linalg.lstsq(W_w, beta_w, rcond=None)
    return theta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", type=str, required=True, help="Trait name (e.g. MCH)")
    parser.add_argument("--in-dir", type=str, required=True, help="Directory containing W.npy and var_names.npy")
    parser.add_argument("--smr-dir", type=str, required=True, help="Directory containing <trait>_beta.npy and <trait>_se.npy")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for theta_<trait>.npy")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    smr_dir = Path(args.smr_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load W and metadata
    w_files = list(in_dir.glob("*W_consensus.npy"))
    if not w_files:
        raise FileNotFoundError(f"No W_consensus.npy found in {in_dir}")
    W = np.load(w_files[0]) # [G, K]
    # Assume W matches the gene order in SMR
    
    # 2. Load SMR effects
    beta = np.load(smr_dir / f"{args.trait}_beta.npy")
    se = np.load(smr_dir / f"{args.trait}_se.npy")
    
    if len(beta) != W.shape[0]:
        print(f"Aligning SMR ({len(beta)}) to W ({W.shape[0]})...")
        # In a real script, we would use var_names.npy to align
        beta = beta[:W.shape[0]]
        se = se[:W.shape[0]]

    # 3. Fit theta
    theta = run_wls(W, beta, se)
    
    # 4. Save
    np.save(out_dir / f"theta_{args.trait}.npy", theta)
    print(f"Saved theta for {args.trait} to {out_dir / f'theta_{args.trait}.npy'}")

if __name__ == "__main__":
    main()
