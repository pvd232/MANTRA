# src/mantra/eggfm/train_energy.py

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import scanpy as sc

from mantra.eggfm.models import EnergyMLP
from mantra.eggfm.dataset import AnnDataExpressionDataset
from mantra.eggfm.config import EnergyModelConfig, EnergyTrainConfig, EnergyModelBundle

def train_energy_model(
    ad_prep: sc.AnnData,         # output of prep(ad, params)
    model_cfg: EnergyModelConfig,
    train_cfg: EnergyTrainConfig,
) -> EnergyModelBundle:
    """
    Train an energy-based model on preprocessed AnnData using denoising score matching.

    Returns an EnergyModelBundle containing:
      - model
      - mean/std used for normalization
      - feature_names (e.g. gene IDs)
      - space ("hvg" by default)
    """
    # -------- device --------
    device = train_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -------- dataset: HVG or PCA --------
    latent_space = "hvg"  # if you want, promote to config later
    if latent_space == "hvg":
        X = ad_prep.X
    else:
        if "X_pca" not in ad_prep.obsm:
            sc.pp.pca(ad_prep, n_comps=50)
        X = ad_prep.obsm["X_pca"]

    dataset = AnnDataExpressionDataset(X)
    n_genes = dataset.X.shape[1]

    # record normalization used by the dataset
    mean = dataset.mean.squeeze(0)  # [D]
    std = dataset.std.squeeze(0)    # [D]
    feature_names = np.array(ad_prep.var_names)

    # -------- model --------
    hidden_dims = tuple(model_cfg.hidden_dims)
    model = EnergyMLP(
        n_genes=n_genes,
        hidden_dims=hidden_dims,
    ).to(device)

    # -------- training hyperparams --------
    batch_size = train_cfg.batch_size
    num_epochs = train_cfg.num_epochs
    lr = train_cfg.lr
    sigma = train_cfg.sigma
    weight_decay = train_cfg.weight_decay
    grad_clip = train_cfg.grad_clip

    early_stop_patience = train_cfg.early_stop_patience  # 0 = off
    early_stop_min_delta = train_cfg.early_stop_min_delta

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    best_state_dict = None
    epochs_without_improve = 0

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for xb in loader:
            xb = xb.to(device)  # (B, D) normalized

            # Sample Gaussian noise
            eps = torch.randn_like(xb)
            y = xb + sigma * eps
            y.requires_grad_(True)

            # Energy and score
            energy = model(y)          # (B,)
            energy_sum = energy.sum()  # scalar

            (grad_y,) = torch.autograd.grad(
                energy_sum,
                y,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            s_theta = -grad_y

            # DSM target: -(y - x) / sigma^2
            target = -(y - xb) / (sigma**2)

            # MSE over batch and dimensions
            loss = ((s_theta - target) ** 2).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(dataset)
        print(
            f"[Energy DSM] Epoch {epoch+1}/{num_epochs}  loss={epoch_loss:.6e}",
            flush=True,
        )

        # ---- early stopping bookkeeping ----
        improved = epoch_loss + early_stop_min_delta < best_loss
        if improved:
            best_loss = epoch_loss
            best_state_dict = model.state_dict()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
            print(
                f"[Energy DSM] Early stopping at epoch {epoch+1} "
                f"(best_loss={best_loss:.6e})",
                flush=True,
            )
            break

    # Restore best weights if we tracked them
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    bundle = EnergyModelBundle(
        model=model,
        mean=mean,
        std=std,
        feature_names=feature_names,
        space=latent_space,
    )
    return bundle
