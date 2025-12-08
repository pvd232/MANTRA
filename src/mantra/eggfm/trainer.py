# src/mantra/eggfm/trainer.py

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import torch
import scanpy as sc
from torch import optim
from torch.utils.data import DataLoader

from mantra.eggfm.models import EnergyMLP
from mantra.eggfm.dataset import AnnDataExpressionDataset
from mantra.eggfm.config import EnergyModelConfig, EnergyTrainConfig, EnergyModelBundle


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
            lr=float(train_cfg.lr),          # force-cast in case YAML gave a string
            weight_decay=float(train_cfg.weight_decay),
        )
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_trait_state: Optional[Dict[str, torch.Tensor]] = None

        self.best_loss: float = float("inf")
        self.best_state_dict: Optional[dict] = None

    def train(self) -> EnergyMLP:
        sigma = float(self.train_cfg.sigma)
        grad_clip = float(self.train_cfg.grad_clip)
        early_stop_patience = int(self.train_cfg.early_stop_patience)
        early_stop_min_delta = float(self.train_cfg.early_stop_min_delta)
        num_epochs = int(self.train_cfg.num_epochs)

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


# --------------------------------------------------------------------
# High-level convenience wrapper: AnnData -> EnergyModelBundle
# --------------------------------------------------------------------

def train_energy_model(
    ad_prep: sc.AnnData,
    model_cfg: EnergyModelConfig,
    train_cfg: EnergyTrainConfig,
    latent_space: str = "hvg",
) -> EnergyModelBundle:
    """
    Convenience wrapper used by scripts:
    AnnData -> AnnDataExpressionDataset -> EnergyMLP -> EnergyTrainer

    `latent_space` controls which representation we train on:
      - "hvg": use ad_prep.X  (HVG log-normalized expression)
      - "pca": run PCA on HVG expression and train in that latent space
      - (other strings can be added later as needed)
    """
    from scipy import sparse as sp_sparse
    from sklearn.decomposition import PCA

    # -------- pick feature space X_feat and embed_meta (f) --------
    # Always start from HVG expression matrix in this AnnData view.
    X_raw = ad_prep.X
    if sp_sparse.issparse(X_raw):
        X_raw = X_raw.toarray()
    X_raw = np.asarray(X_raw, dtype=np.float32)  # (N, G)

    if latent_space == "hvg":
        # Identity feature map: f(x) = x
        X_feat = X_raw  # (N, G)
        embed_meta: Dict[str, Any] = {
            "type": "identity",
        }

    elif latent_space == "pca":
        # PCA latent: f(x) = (x - μ_pca) @ V_pca^T
        # Number of components: from train_cfg if available, else default to 20.
        n_components = getattr(train_cfg, "latent_n_components", 20)
        print(
            f"[EGGFM trainer] Using PCA latent space with n_components={n_components}",
            flush=True,
        )
        pca = PCA(n_components=n_components, random_state=getattr(train_cfg, "seed", 0))
        X_feat = pca.fit_transform(X_raw).astype(np.float32)  # (N, D_pca)

        embed_meta = {
            "type": "pca",
            "mean": pca.mean_.astype(np.float32),         # (G,)
            "components": pca.components_.astype(np.float32),  # (D_pca, G)
            "n_components": int(n_components),
        }

    else:
        raise ValueError(
            f"Unsupported latent_space={latent_space!r}. "
            "Use 'hvg' or 'pca' for now."
        )

    # -------- dataset in the chosen feature space --------
    dataset = AnnDataExpressionDataset(X_feat)
    n_genes = dataset.X.shape[1]  # feature dim = D (HVG or PCA)

    # record normalization (always in the *model feature space*)
    mean = dataset.mean  # [D]
    std = dataset.std    # [D]

    # For both HVG and PCA we keep the gene names (for alignment later).
    feature_names = np.array(ad_prep.var_names)

    # -------- model --------
    hidden_dims = tuple(model_cfg.hidden_dims)
    model = EnergyMLP(
        n_genes=n_genes,
        hidden_dims=hidden_dims,
    )

    # -------- trainer --------
    trainer = EnergyTrainer(
        model=model,
        dataset=dataset,
        train_cfg=train_cfg,
    )
    best_model = trainer.train()

    return EnergyModelBundle(
        model=best_model,
        mean=mean,
        std=std,
        feature_names=feature_names,
        space=latent_space,
        embed_meta=embed_meta,
    )
