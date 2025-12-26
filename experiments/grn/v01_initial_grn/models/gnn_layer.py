from __future__ import annotations

import torch
from torch import nn, Tensor


class GeneGNNLayer(nn.Module):
    """
    Single message-passing layer with FiLM conditioning on global cond vector.

    Inputs:
        h:    [B, G, d_in]  node features
        cond: [B, d_cond]   global condition (reg ± dose)
        A:    [G, G]        (row- or sym-normalized adjacency)
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_cond: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.cond_to_film = nn.Linear(d_cond, 2 * d_out)
        self.norm = nn.LayerNorm(d_out)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: Tensor,        # [B, G, d_in]
        cond: Tensor,     # [B, d_cond]
        A: Tensor,        # [G, G]
    ) -> Tensor:
        # Message passing: (G,G) x (B,G,d_in) -> (B,G,d_in)
        agg = torch.einsum("ij,bjd->bid", A, h)  # [B, G, d_in]
        h_lin = self.linear(agg)                 # [B, G, d_out]

        # FiLM from global condition
        gamma_beta = self.cond_to_film(cond)     # [B, 2*d_out]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, d_out] each
        gamma = gamma.unsqueeze(1)               # [B, 1, d_out]
        beta = beta.unsqueeze(1)                 # [B, 1, d_out]        
        h_film = gamma * h_lin + beta
        h_norm = self.norm(h_film)
        out = self.act(h_norm)
        out = self.dropout(out)
        return out
