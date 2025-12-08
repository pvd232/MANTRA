#!/usr/bin/env python
# src/mantra/embeddings/config.py
"""
Configuration and summary containers for manifold embeddings on HVGs.

EmbeddingConfig:
  - hyperparameters for PCA / DiffMap / UMAP / PHATE / X_hvg_trunc
EmbeddingSummary:
  - lightweight record of which embeddings were computed and their shapes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List


@dataclass
class EmbeddingConfig:
    """
    Hyperparameters for computing manifold embeddings on a QC’d AnnData with HVGs.

    Mirrors `scripts/hvg_embed.py` so you can drive that script entirely from
    a YAML block (e.g. `embeddings:` in configs/params.yml).
    """

    # ---- Core embedding geometry ----
    n_components: int = 20      # target embedding dimension (PCA, diffmap, UMAP, PHATE)
    n_neighbors: int = 30       # k for kNN-based methods
    seed: int = 0               # RNG seed for PCA/UMAP/neighbors

    # ---- HVG usage ----
    use_hvg_only: bool = True       # subset to HVGs if ad.var[hvg_key] present
    hvg_key: str = "highly_variable"
    hvg_trunc: Optional[int] = None # size of X_hvg_trunc; None -> fall back to n_components

    # ---- Which methods to run ----
    run_hvg_trunc: bool = True  # compute X_hvg_trunc in .obsm
    run_pca: bool = True        # compute X_pca
    run_diffmap: bool = True    # compute X_diffmap (on top of X_pca)
    run_umap: bool = True       # compute X_umap (on top of X_pca)
    run_phate: bool = True      # compute X_phate (if phate is installed)


@dataclass
class EmbeddingSummary:
    """
    Lightweight record of which embeddings were written to `ad.obsm`
    and their shapes.

    This is optional but useful for logging / manifest generation.
    """

    # Names of `.obsm` keys that were successfully written
    obsm_keys: List[str]  # e.g. ["X_hvg_trunc", "X_pca", "X_diffmap", "X_umap", "X_phate"]

    # Map from obsm key -> (n_cells, dim)
    shapes: Dict[str, Tuple[int, int]]

    # Config used to generate these embeddings
    config: EmbeddingConfig
