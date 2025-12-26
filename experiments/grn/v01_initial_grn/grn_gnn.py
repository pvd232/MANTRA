from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor

from .condition_encoder import ConditionEncoder
from .gnn_layer import GeneGNNLayer


class GRNGNN(nn.Module):
    """
    f_theta: (reg_idx, dose?) -> ΔE_pred per gene, conditioned on gene graph.

    Forward:
        reg_idx: [B]     (long)
        dose:    [B] or None
        A:       [G, G]  (normalized adjacency for genes)

        returns ΔE_pred: [B, G]
    """
    def __init__(
        self,
        n_regulators: int,
        n_genes: int,
        n_layers: int = 3,
        gene_emb_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_dose: bool = True,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.use_dose = use_dose

        # Global condition encoder (reg ± dose)
        self.cond_encoder = ConditionEncoder(
            n_regulators=n_regulators,
            hidden_dim=hidden_dim,
            reg_dim=hidden_dim // 2,
            dose_dim=hidden_dim // 4,
            use_dose=use_dose,
        )

        # Learnable per-gene initial embeddings h_g^(0)
        self.gene_emb = nn.Parameter(
            0.01 * torch.randn(n_genes, gene_emb_dim)
        )

        # Stack of FiLM-conditioned GNN layers
        layers = []
        d_in = gene_emb_dim
        for _ in range(n_layers):
            layers.append(
                GeneGNNLayer(
                    d_in=d_in,
                    d_out=hidden_dim,
                    d_cond=hidden_dim,
                    dropout=dropout,
                )
            )
            d_in = hidden_dim
        self.layers = nn.ModuleList(layers)

        # Per-gene readout → scalar ΔE_pred
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        reg_idx: Tensor,           # [B]
        dose: Optional[Tensor],    # [B] or None
        A: Tensor,                 # [G, G]
    ) -> Tensor:
        cond = self.cond_encoder(
            reg_idx,
            dose if self.use_dose else None,
        )  # [B, hidden_dim]

        B = reg_idx.shape[0]
        # Broadcast gene embeddings across batch: [B, G, gene_emb_dim]
        h = self.gene_emb.unsqueeze(0).expand(B, self.n_genes, -1)

        # GNN layers
        for layer in self.layers:
            h = layer(h, cond, A)  # [B, G, hidden_dim]

        # Per-gene linear head → ΔE_pred
        delta_e = self.readout(h).squeeze(-1)   # [B, G]
        return delta_e
