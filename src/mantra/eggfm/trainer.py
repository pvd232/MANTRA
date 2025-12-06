# src/mantra/eggfm/trainer.py

from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from mantra.eggfm.models import EnergyMLP
from mantra.eggfm.dataset import AnnDataExpressionDataset
from mantra.eggfm.config import EnergyTrainConfig

class EnergyTrainer:
    """
    Denoising score-matching trainer for EnergyMLP.

    Given:
      - model
      - standardized dataset (AnnDataExpressionDataset)
      - EnergyTrainConfig

    It runs the DSM loop and returns the best-trained model.
    """

    def __init__(
        self,
        model: EnergyMLP,
        dataset: AnnDataExpressionDataset,
        train_cfg: EnergyTrainConfig,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.train_cfg = train_cfg

        device_str = train_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.model.to(self.device)

        self.loader = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )

        self.best_loss: float = float("inf")
        self.best_state_dict: Optional[dict] = None

    def train(self) -> EnergyMLP:
        sigma = self.train_cfg.sigma
        grad_clip = self.train_cfg.grad_clip
        early_stop_patience = self.train_cfg.early_stop_patience
        early_stop_min_delta = self.train_cfg.early_stop_min_delta
        num_epochs = self.train_cfg.num_epochs

        self.model.train()
        epochs_without_improve = 0

        n_total = len(self.dataset)

        for epoch in range(num_epochs):
            running_loss = 0.0

            for xb in self.loader:
                xb = xb.to(self.device)  # (B, D), already standardized

                # Sample Gaussian noise
                eps = torch.randn_like(xb)
                y = xb + sigma * eps
                y.requires_grad_(True)

                # Energy and score
                energy = self.model(y)          # (B,)
                energy_sum = energy.sum()       # scalar

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

                self.optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        grad_clip,
                    )
                self.optimizer.step()

                running_loss += loss.item() * xb.size(0)

            epoch_loss = running_loss / n_total
            print(
                f"[Energy DSM] Epoch {epoch+1}/{num_epochs}  loss={epoch_loss:.6e}",
                flush=True,
            )

            improved = epoch_loss + early_stop_min_delta < self.best_loss
            if improved:
                self.best_loss = epoch_loss
                self.best_state_dict = self.model.state_dict()
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
                print(
                    f"[Energy DSM] Early stopping at epoch {epoch+1} "
                    f"(best_loss={self.best_loss:.6e})",
                    flush=True,
                )
                break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self.model
