#!/usr/bin/env python
# src/mantra/grn/config.py
"""
Configuration and result containers for the GRN GNN.

GRNModelConfig:
  - architecture choices for GRNGNN + optional trait head
GRNTrainConfig:
  - training loop and optimizer hyperparameters
GRNLossConfig:
  - lambda weights for composite loss terms
GRNResults:
  - minimal container for training history and best-checkpoint metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GRNModelConfig:
    """
    Architecture config for the GRN GNN (GRNGNN).

    Covers the number of message-passing layers, embedding widths, dropout,
    and whether to include dose inputs and a trait head.
    """

    # Core GNN
    n_layers: int = 3          # number of message-passing layers
    gene_emb_dim: int = 64     # dimensionality of per-gene embeddings
    hidden_dim: int = 128      # hidden width inside GNN layers
    dropout: float = 0.1       # dropout rate applied in GNN layers

    # Input augmentation
    use_dose: bool = False     # if True, model expects a dose covariate per sample

    # Optional trait head on top of program representation
    n_traits: int = 0          # 0 = no trait head; >0 = predict this many traits
    trait_hidden_dim: int = 64 # hidden width inside the trait head MLP


@dataclass
class GRNTrainConfig:
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_epochs: int = 50
    grad_clip: float = 1.0
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.0

    # NEW: cosine LR schedule
    use_cosine_lr: bool = False
    cosine_eta_min: float = 1e-6  # minimum LR



@dataclass
class GRNLossConfig:
    """
    Lambda weights for each loss term in GRN training.

    All lambdas are >= 0; setting a lambda to 0.0 effectively disables that term.
    """

    lambda_geo: float = 0.0    # geometry prior term (EGGFM energy)
    lambda_prog: float = 0.0   # program-level term (ΔP_obs via W)
    lambda_trait: float = 0.0  # trait-level supervision term


@dataclass
class GRNResults:
    """
    Container for GRN training history and best-checkpoint metadata.

    This is intentionally minimal and can be extended as needed with
    per-loss curves (expr/geo/prog/trait) or additional diagnostics.
    """

    # Training curves
    train_loss: np.ndarray                 # shape: [n_epochs]
    val_loss: Optional[np.ndarray] = None  # shape: [n_epochs] or None if no val

    # Index of epoch with best validation loss (or final epoch if no val)
    best_epoch: int = -1

    # Configs used for this run (useful for manifesting)
    model_cfg: GRNModelConfig = GRNModelConfig()
    train_cfg: GRNTrainConfig = GRNTrainConfig()
    loss_cfg: GRNLossConfig = GRNLossConfig()
