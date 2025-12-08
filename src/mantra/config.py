# src/mantra/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, List

import numpy as np
from torch import nn


# =======================
# EGGFM / energy configs
# =======================

@dataclass
class EnergyModelConfig:
    """
    Architecture config for EnergyMLP (or any energy model).
    """
    hidden_dims: Sequence[int] = (512, 512, 512, 512)


@dataclass
class EnergyTrainConfig:
    """
    Hyperparameters for DSM energy training.
    """
    batch_size: int = 2048
    num_epochs: int = 50
    lr: float = 1e-4
    sigma: float = 0.1
    weight_decay: float = 0.0
    grad_clip: float = 0.0

    early_stop_patience: int = 0        # 0 = off
    early_stop_min_delta: float = 0.0

    device: Optional[str] = None
    n_cells_sample: Optional[int] = None
    max_hvg: Optional[int] = None


@dataclass
class EnergyModelBundle:
    """
    Container for a trained energy model + normalization metadata.
    """
    model: nn.Module                   # keep this generic to avoid imports
    mean: np.ndarray                   # [D]
    std: np.ndarray                    # [D]
    feature_names: Optional[List[str]] = None  # e.g. gene IDs
    space: str = "hvg"                 # 'hvg', 'pca', 'embedding', ...


# =======================
# GRN configs
# =======================

@dataclass
class GRNModelConfig:
    """
    Architecture config for the GRN GNN (GRNGNN).
    """
    n_layers: int = 3
    gene_emb_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.1
    use_dose: bool = False

    # trait head (optional)
    n_traits: int = 0
    trait_hidden_dim: int = 64


@dataclass
class GRNTrainConfig:
    """
    Hyperparameters for GRN training.
    """
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 50
    grad_clip: float = 0.0
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0


@dataclass
class GRNLossConfig:
    """
    Lambda weights for each loss term in GRN training.
    """
    lambda_geo: float = 0.0
    lambda_prog: float = 0.0
    lambda_trait: float = 0.0

# =======================
# cNMF configs
# =======================

@dataclass
class CNMFConfig:
    """
    Hyperparameters and preprocessing options for consensus NMF on an AnnData object.

    This is intentionally explicit + typed so experiments are reproducible and
    can be reconstructed from a single YAML block.
    """

    # ---- Core factorization ----
    n_components: int = 75           # K = number of programs
    n_restarts: int = 20             # R = number of independent NMF runs
    max_iter: int = 400              # per-run NMF iterations
    tol: float = 1e-4                # convergence tolerance

    # ---- Data / preprocessing ----
    use_hvg_only: bool = True        # restrict to HVGs if available
    hvg_key: str = "highly_variable" # column in ad.var marking HVGs
    n_top_genes: int = 3000          # top HVGs to keep for cNMF
    min_cells_per_gene: int = 10     # drop genes expressed in < this many cells
    scale_cells: bool = True         # per-cell L1 / library-size normalization
    scale_genes: bool = False        # optional gene-wise standardization (usually False)

    # ---- NMF regularization (if supported by the backend) ----
    alpha: float = 0.0               # overall regularization strength
    l1_ratio: float = 0.0            # 0 = pure L2, 1 = pure L1

    # ---- Consensus clustering ----
    consensus_method: str = "kmeans" # currently only "kmeans" implemented
    consensus_kmeans_n_init: int = 10

    # Stable-program filtering
    filter_by_coverage: bool = True  # drop unstable programs if True
    min_run_coverage: float = 0.7    # keep programs present in ≥ this fraction of runs

    # ---- Reproducibility ----
    random_state: int = 7            # master RNG seed

@dataclass
class CNMFResults:
    """
    Container for consensus NMF results.
    All arrays are in the filtered gene space used for cNMF.
    """

    # Factorization
    W: np.ndarray                    # [G, K] per-run-averaged or stacked programs (optional)
    H: np.ndarray                    # [K, N] usage matrix (if you keep it)

    # Consensus summary
    W_consensus: np.ndarray          # [G, K_stable] final stable program loadings
    W_consensus_full: np.ndarray     # [G, K] before filtering (for diagnostics)
    cluster_assignments: np.ndarray  # [R * K] cluster ID per run-program
    run_coverage: np.ndarray         # [K] fraction of runs each consensus program appears in

    # Metadata
    gene_names: np.ndarray           # [G] gene IDs for the rows of W
    config: CNMFConfig               # config used to generate these results

# =======================
# Embedding configs
# =======================  
@dataclass
class EmbeddingConfig:
    """
    Hyperparameters for computing manifold embeddings on a QC’d AnnData with HVGs.
    Mirrors scripts/hvg_embed.py so you can drive it entirely from YAML.
    """

    # Core embedding geometry
    n_components: int = 20          # target embedding dimension
    n_neighbors: int = 30           # k for kNN-based methods (diffmap, UMAP, PHATE, etc.)
    seed: int = 0                   # RNG seed

    # HVG usage
    use_hvg_only: bool = True       # subset to HVGs if ad.var[hvg_key] is present
    hvg_key: str = "highly_variable"
    hvg_trunc: Optional[int] = None # if set, size of X_hvg_trunc; else fall back to n_components

    # Which methods to run
    run_hvg_trunc: bool = True      # compute X_hvg_trunc
    run_pca: bool = True            # compute X_pca
    run_diffmap: bool = True        # compute X_diffmap
    run_umap: bool = True           # compute X_umap
    run_phate: bool = True          # compute X_phate if phate is installed

    # Legacy compat: if True, skip others and just do HVG trunc + PHATE
    only_hvg_phate: bool = False