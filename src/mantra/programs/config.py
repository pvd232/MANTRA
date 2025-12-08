#!/usr/bin/env python
# src/mantra/programs/config.py
"""
Configuration and result containers for consensus NMF (cNMF) on AnnData.

CNMFConfig:
  - hyperparameters and preprocessing options (HVG, filters, scaling, NMF, consensus)
CNMFResults:
  - minimal container for consensus W and diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CNMFConfig:
    """
    Hyperparameters and preprocessing options for consensus NMF on an AnnData object.

    This is intentionally small and explicit so experiments are reproducible and easy
    to reconstruct from a single YAML block or CLI + defaults.
    """

    # ---- Core factorization ----
    n_components: int = 75        # K = number of programs
    n_restarts: int = 20          # R = number of independent NMF runs
    max_iter: int = 400           # per-run NMF iterations
    tol: float = 1e-4             # convergence tolerance

    # ---- Data / preprocessing ----
    use_hvg_only: bool = True         # restrict to HVGs if available
    hvg_key: str = "highly_variable"  # column in ad.var marking HVGs
    min_cells_per_gene: int = 10      # drop genes expressed in < this many cells
    scale_cells: bool = True          # per-cell library-size scaling (L1)

    # ---- NMF regularization ----
    alpha: float = 0.0           # L1/L2 regularization strength
    l1_ratio: float = 0.0        # 0 = pure L2, 1 = pure L1

    # ---- Consensus clustering ----
    consensus_kmeans_n_init: int = 10   # KMeans restarts

    # Stable-program filtering
    filter_by_coverage: bool = True     # drop unstable programs if True
    min_run_coverage: float = 0.7       # keep programs present in ≥ this fraction of runs

    # ---- Reproducibility ----
    random_state: int = 7          # master RNG seed


@dataclass
class CNMFResults:
    """
    Container for consensus NMF results.
    All arrays are in the filtered gene space used for cNMF.
    """

    # Consensus programs
    W_consensus: np.ndarray          # [G, K_stable] final stable program loadings
    programs_all: np.ndarray         # [R*K, G] all per-run programs (L2-normalized)
    cluster_assignments: np.ndarray  # [R*K] cluster ID per run-program
    run_coverage: np.ndarray         # [K] fraction of runs each cluster appears in

    # Metadata
    gene_names: np.ndarray           # [G] gene IDs for the rows of W
    cell_idx: np.ndarray             # indices of cells used
    config: CNMFConfig               # config used to generate these results
