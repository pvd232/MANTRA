# src/mantra/grn/priors.py
from __future__ import annotations

from typing import Sequence, Optional, Dict, Any

import numpy as np
import torch
from torch import nn

from mantra.eggfm.models import EnergyMLP


class EggfmEnergyPrior(nn.Module):
    """
    Wraps an EnergyMLP and its feature-space stats + embedding map f(x).

    Forward contract:
      - Input: x_gene [B, G] in gene space, ordered according to var_names.
      - Internally:
          * if embed_meta['type'] == 'identity':
                z = x_gene
          * if embed_meta['type'] == 'pca':
                z = (x_gene - μ_pca) @ V_pca^T
          * standardize: z_std = (z - mean) / std
          * energy = energy_model(z_std)
      - Output: energy [B] (higher = less likely)
    """

    def __init__(
        self,
        energy_model: EnergyMLP,
        mean: np.ndarray,
        std: np.ndarray,
        space: str,
        embed_meta: Dict[str, Any],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.energy_model = energy_model.to(device)
        self.space = space

        # register buffers for normalization in feature space
        mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
        std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
        self.register_buffer("feat_mean", mean_t)
        self.register_buffer("feat_std", std_t)

        # embedding metadata: describes f(x)
        self.embed_type = embed_meta.get("type", "identity")

        if self.embed_type == "pca":
            pca_mean = np.asarray(embed_meta["mean"], dtype=np.float32)   # [G]
            pca_components = np.asarray(embed_meta["components"], dtype=np.float32)  # [D_pca, G]
            self.register_buffer(
                "pca_mean",
                torch.as_tensor(pca_mean, dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "pca_components",
                torch.as_tensor(pca_components, dtype=torch.float32, device=device),
            )
        else:
            # identity map: f(x) = x
            self.pca_mean = None
            self.pca_components = None

    def forward(self, x_gene: torch.Tensor) -> torch.Tensor:
        """
        x_gene: [B, G] in the gene space defined by ckpt['var_names'].
        Returns: energy [B]
        """
        if self.embed_type == "pca":
            # x_gene -> PCA latent
            # x_gene: [B, G], pca_mean: [G], pca_components: [D, G]
            x_centered = x_gene - self.pca_mean  # broadcast: [B, G]
            z = torch.matmul(x_centered, self.pca_components.t())  # [B, D]
        else:
            # identity: f(x) = x
            z = x_gene

        # normalize in feature space
        z_std = (z - self.feat_mean) / (self.feat_std + 1e-8)

        # EnergyMLP expects [B, D] and returns [B]
        energy = self.energy_model(z_std)
        return energy


def build_energy_prior_from_ckpt(
    ckpt_path: str,
    gene_names: Sequence[str],
    device: torch.device,
) -> EggfmEnergyPrior:
    """
    Build an EggfmEnergyPrior from a saved EGGFM checkpoint.

    - ckpt['space'] ∈ {'hvg', 'pca'}
    - ckpt['var_names'] are the gene IDs that define the x_gene coordinate system
    - ckpt['embed_meta'] describes the embedding f(x) used during training
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    space = ckpt.get("space", "hvg")
    embed_meta = ckpt.get("embed_meta", {"type": "identity"})
    hidden_dims = tuple(ckpt["model_cfg"]["hidden_dims"])
    n_genes_feat = int(ckpt["n_genes"])  # feature-dim inside EnergyMLP

    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)

    # Rebuild EnergyMLP in the feature space dimension
    energy_model = EnergyMLP(
        n_genes=n_genes_feat,
        hidden_dims=hidden_dims,
    )
    energy_model.load_state_dict(ckpt["state_dict"])

    # Optional sanity check: gene_names alignment
    ckpt_var_names = np.array(ckpt["var_names"], dtype=str)
    if ckpt_var_names.shape[0] != len(gene_names):
        raise ValueError(
            f"Energy checkpoint var_names has {ckpt_var_names.shape[0]} genes, "
            f"but gene_names passed in has {len(gene_names)}."
        )

    return EggfmEnergyPrior(
        energy_model=energy_model,
        mean=mean,
        std=std,
        space=space,
        embed_meta=embed_meta,
        device=device,
    )
