#!/usr/bin/env python
# src/mantra/eggfm/config.py
"""
Configuration containers for EGGFM (Energy-Guided Geometric Flow Model).

EnergyModelConfig:
  - architecture hyperparameters for the energy network (EnergyMLP)
EnergyTrainConfig:
  - DSM training hyperparameters and subset options
EnergyModelBundle:
  - trained model + normalization metadata for downstream use
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, List

import numpy as np
from torch import nn


@dataclass
class EnergyModelConfig:
    """
    Architecture config for the energy network (e.g. EnergyMLP).

    This is deliberately small and explicit so that the model can be fully
    reconstructed from a YAML block or CLI + defaults.
    """

    # Hidden layer widths for the MLP (input/output inferred from data)
    hidden_dims: Sequence[int] = (512, 512, 512, 512)


@dataclass
class EnergyTrainConfig:
    """
    Hyperparameters for denoising score matching (DSM) training of EGGFM.

    Controls batch size, LR, DSM noise scale, regularization, and optional
    subsampling of cells / HVGs.
    """

    # Core training loop
    batch_size: int = 2048
    num_epochs: int = 50
    lr: float = 1e-4

    # DSM noise scale (Gaussian corruption std)
    sigma: float = 0.1

    # Regularization / stabilization
    weight_decay: float = 0.0
    grad_clip: float = 0.0  # 0.0 = disabled

    # Early stopping on DSM loss (0 patience = disabled)
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0

    # Device + subsampling
    device: Optional[str] = None          # e.g. "cuda", "cpu", or None -> auto
    n_cells_sample: Optional[int] = None  # if set, sample this many cells per epoch
    max_hvg: Optional[int] = None         # if set, restrict to top-N HVGs for EGGFM


@dataclass
class EnergyModelBundle:
    """
    Container for a trained energy model plus normalization metadata.

    This is what `train_energy.py` should save/load and what downstream
    consumers (GRN, cNMF alignment, NPZ construction) can depend on.
    """

    # Trained energy model (kept generic as nn.Module)
    model: nn.Module

    # Per-feature normalization statistics in the training space
    mean: np.ndarray          # shape: [D]
    std: np.ndarray           # shape: [D]

    # Optional metadata
    feature_names: Optional[List[str]] = None  # gene IDs or embedding feature names
    space: str = "hvg"                         # e.g. "hvg", "X_pca", "embedding"
