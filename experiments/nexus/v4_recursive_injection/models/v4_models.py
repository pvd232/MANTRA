import torch
from torch import nn, Tensor
from typing import Optional, Dict

from mantra.grn.models import ConditionEncoder, GeneGNNLayer

class GeneGNNLayer_V4(nn.Module):
    """
    FiLM-conditioned GNN layer with Nexus Recursive Injection.
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_cond: int,
        d_nexus: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.cond_to_film = nn.Linear(d_cond, 2 * d_out)
        self.nexus_to_gate = nn.Sequential(
            nn.Linear(d_nexus, d_out),
            nn.Sigmoid()
        )
        self.nexus_to_shift = nn.Linear(d_nexus, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: Tensor,        # [B, G, d_in]
        cond: Tensor,     # [B, d_cond]
        A: Tensor,        # [G, G]
        nexus_signal: Optional[Tensor] = None, # [B, d_nexus]
    ) -> Tensor:
        # Message passing
        agg = torch.einsum("ij,bjd->bid", A, h)
        h_lin = self.linear(agg)

        # Baseline FiLM
        gamma_beta = self.cond_to_film(cond) # [B, 2*d_out]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        
        # Recursive Nexus Injection
        if nexus_signal is not None:
            gate = self.nexus_to_gate(nexus_signal).unsqueeze(1) # [B, 1, d_out]
            shift = self.nexus_to_shift(nexus_signal).unsqueeze(1) # [B, 1, d_out]
            # Influence the baseline FiLM parameters or the hidden state directly
            # Here: Gated residual shift
            h_lin = h_lin + gate * shift

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        h_film = gamma * h_lin + beta
        h_norm = self.norm(h_film)
        out = self.act(h_norm)
        out = self.dropout(out)
        return out

class GRNGNN_V4(nn.Module):
    """
    MANTRA GNN with Recursive Nexus Injection support.
    """
    def __init__(
        self,
        n_regulators: int,
        n_genes: int,
        n_layers: int = 3,
        gene_emb_dim: int = 64,
        hidden_dim: int = 128,
        d_nexus: int = 256, # Nexus hidden state size
        dropout: float = 0.1,
        use_dose: bool = False,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.use_dose = use_dose

        self.cond_encoder = ConditionEncoder(
            n_regulators=n_regulators,
            hidden_dim=hidden_dim,
            reg_dim=hidden_dim // 2,
            dose_dim=hidden_dim // 4,
            use_dose=use_dose,
        )

        self.gene_emb = nn.Parameter(0.01 * torch.randn(n_genes, gene_emb_dim))

        layers = []
        d_in = gene_emb_dim
        for _ in range(n_layers):
            layers.append(
                GeneGNNLayer_V4(
                    d_in=d_in,
                    d_out=hidden_dim,
                    d_cond=hidden_dim,
                    d_nexus=d_nexus,
                    dropout=dropout,
                )
            )
            d_in = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        reg_idx: Tensor,
        dose: Optional[Tensor],
        A: Tensor,
        nexus_signal: Optional[Tensor] = None, # [B, d_nexus]
    ) -> Tensor:
        cond = self.cond_encoder(reg_idx, dose if self.use_dose else None)
        B = reg_idx.shape[0]
        h = self.gene_emb.unsqueeze(0).expand(B, self.n_genes, -1)

        for layer in self.layers:
            h = layer(h, cond, A, nexus_signal=nexus_signal)

        delta_e = self.readout(h).squeeze(-1)
        return delta_e
