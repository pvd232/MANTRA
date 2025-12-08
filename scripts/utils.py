#!/usr/bin/env python3
# src/mantra/eggfm/utils.py
"""
Utility helpers for EGGFM training and experiments.

Currently provides:
  - subset_anndata: random subsampling of AnnData rows (cells)
    with a fixed random seed for reproducibility.
"""

import numpy as np
import scanpy as sc


def subset_anndata(ad: sc.AnnData, n_cells_sample: int, random_state: int) -> sc.AnnData:
    """
    Randomly subset AnnData rows (cells).
    If n_cells_sample >= n_obs or n_cells_sample <= 0, returns ad.copy().
    """
    n = ad.n_obs
    m = min(int(n_cells_sample), n)
    if m <= 0 or m == n:
        return ad.copy()

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=m, replace=False)
    return ad[idx].copy()