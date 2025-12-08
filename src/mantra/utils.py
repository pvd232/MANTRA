# src/mantra/utils.py
from __future__ import annotations

from typing import Any
import numpy as np
import scanpy as sc


def subset_anndata(
    ad: sc.AnnData,
    n_cells_sample: int,
    random_state: int,
) -> sc.AnnData:
    """
    Randomly subset AnnData rows (cells).

    If n_cells_sample >= n_obs or n_cells_sample <= 0, returns ad.copy().

    Parameters
    ----------
    ad
        Input AnnData (cells × genes).
    n_cells_sample
        Desired number of cells to sample.
    random_state
        Seed for the RNG.

    Returns
    -------
    AnnData
        New AnnData with a subset of cells.
    """
    n = ad.n_obs
    m = min(int(n_cells_sample), n)

    if m <= 0 or m == n:
        return ad.copy()

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=m, replace=False)
    return ad[idx].copy()
