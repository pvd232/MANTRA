# src/mantra/smr/readout.py
"""
Logic for mapping gene-program deltas (\Delta P) to phenotypic trait deltas (\Delta y).
Uses Weighted Least Squares (WLS) to fit program-level effect sizes (\theta)
from gene-level SMR/GWAS summary statistics.
"""

import numpy as np
import torch
from pathlib import Path

def fit_theta_wls(beta_gene, se_gene, W):
    """
    Fit \theta using WLS: \theta = (W^T V^{-1} W)^{-1} W^T V^{-1} \beta
    where V is the diagonal matrix of squared standard errors (se_gene^2).
    
    Args:
        beta_gene: Vector of gene-level effect sizes [G]
        se_gene: Vector of gene-level standard errors [G]
        W: Gene-to-program loading matrix [G, K]
        
    Returns:
        theta: Program-level effect sizes [K]
    """
    # Inverse variance weights
    weights = 1.0 / (se_gene**2 + 1e-12)
    
    # Construct WLS components
    # (W.T * diag(w) * W) theta = W.T * diag(w) * beta
    
    # Efficient computation without constructing full diag(V^{-1})
    W_weighted = W * weights[:, np.newaxis]
    lhs = W.T @ W_weighted # [K, K]
    rhs = W_weighted.T @ beta_gene # [K]
    
    # Solve for theta
    theta = np.linalg.solve(lhs, rhs)
    return theta

def load_smr_stats(traits_path: Path, trait_name: str):
    """Load beta and se for a given trait."""
    beta = np.load(traits_path / f"{trait_name}_beta.npy")
    se = np.load(traits_path / f"{trait_name}_se.npy")
    return beta, se
