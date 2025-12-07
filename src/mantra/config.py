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
