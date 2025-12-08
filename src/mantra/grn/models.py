# src/mantra/grn/models.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn, Tensor


# ---------------------------------------------------------------------
# 1. Conditioning encoder (regulator ± dose)
# ---------------------------------------------------------------------

class ConditionEncoder(nn.Module):
    """
    Encodes regulator (always) and optionally dose -> conditioning vector c.

    Shapes:
        reg_idx: [B]  (long)
        dose:    [B] or [B, 1] (float, optional)
        output:  [B, hidden_dim]
    """
    def __init__(
        self,
        n_regulators: int,
        hidden_dim: int = 128,
        reg_dim: int = 64,
        dose_dim: int = 16,
        use_dose: bool = True,
    ) -> None:
        super().__init__()
        self.use_dose = use_dose

        self.reg_embed = nn.Embedding(n_regulators, reg_dim)

        if use_dose:
            self.dose_mlp = nn.Sequential(
                nn.Linear(1, dose_dim),
                nn.ReLU(),
                nn.Linear(dose_dim, dose_dim),
                nn.ReLU(),
            )
            in_dim = reg_dim + dose_dim
        else:
            self.dose_mlp = None
            in_dim = reg_dim

        self.out = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        reg_idx: Tensor,            # [B]
        dose: Optional[Tensor] = None,  # [B] or [B, 1] or None
    ) -> Tensor:
        reg_emb = self.reg_embed(reg_idx)  # [B, reg_dim]

        if self.use_dose:
            if dose is None:
                raise ValueError("dose tensor is required when use_dose=True")
            dose = dose.view(-1, 1)        # [B, 1]
            dose_emb = self.dose_mlp(dose) # [B, dose_dim]
            x = torch.cat([reg_emb, dose_emb], dim=-1)
        else:
            x = reg_emb

        cond = self.out(x)                # [B, hidden_dim]
        return cond


# ---------------------------------------------------------------------
# 2. FiLM-conditioned GNN layer
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# 3. GRN GNN model: (reg, dose?) -> ΔE_pred
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# 4. Optional trait head: ΔP -> Δy
# ---------------------------------------------------------------------

class TraitHead(nn.Module):
    """
    Simple MLP mapping program deltas ΔP -> trait deltas Δy.

    Input:
        deltaP: [B, K]
    Output:
        deltaY: [B, T]
    """
    def __init__(
        self,
        n_programs: int,
        n_traits: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_programs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_traits),
        )

    def forward(self, deltaP: Tensor) -> Tensor:
        return self.net(deltaP)


# ---------------------------------------------------------------------
# 5. Loss configuration + loss computation
# ---------------------------------------------------------------------

@dataclass
class GRNLossConfig:
    """
    Lambda weights for each loss term.
    """
    lambda_geo: float = 0.0
    lambda_prog: float = 0.0
    lambda_trait: float = 0.0

def compute_grn_losses(
    model: GRNGNN,
    A: torch.Tensor,                           # [G, G]
    batch: dict[str, torch.Tensor],
    x_ref: torch.Tensor,                       # [G]
    energy_prior: nn.Module,                      # EnergyScorerPrior
    W: torch.Tensor,                           # [G, K]
    loss_cfg: GRNLossConfig,
    trait_head: Optional[nn.Module] = None,
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device

    reg_idx = batch["reg_idx"].to(device)      # [B]
    deltaE_obs = batch["deltaE"].to(device)    # [B, G]

    dose = batch.get("dose", None)
    if dose is not None:
        dose = dose.to(device)

    A = A.to(device)
    x_ref = x_ref.to(device)
    W = W.to(device)

    # 1) ΔE prediction
    deltaE_pred = model(reg_idx=reg_idx, dose=dose, A=A)  # [B, G]

    # 2) Expression loss
    L_expr = ((deltaE_pred - deltaE_obs) ** 2).mean()

    # 3) Geometric prior (frozen EGGFM)
    x_hat = x_ref.unsqueeze(0) + deltaE_pred   # [B, G]
    energy = energy_prior(x_hat)                  # [B]
    L_geo = loss_cfg.lambda_geo * energy.mean()

    # 4) Program-level supervision
    deltaP_pred = deltaE_pred @ W              # [B, K]

    L_prog = torch.zeros((), device=device)
    if "deltaP_obs" in batch:
        deltaP_obs = batch["deltaP_obs"].to(device)
        L_prog = loss_cfg.lambda_prog * ((deltaP_pred - deltaP_obs) ** 2).mean()

    # 5) Trait head (optional)
    L_trait = torch.zeros((), device=device)
    if trait_head is not None and "deltaY_obs" in batch:
        deltaY_obs = batch["deltaY_obs"].to(device)
        deltaY_pred = trait_head(deltaP_pred)
        L_trait = loss_cfg.lambda_trait * ((deltaY_pred - deltaY_obs) ** 2).mean()

    L_total = L_expr + L_geo + L_prog + L_trait

    return {
        "loss": L_total,
        "L_expr": L_expr.detach(),
        "L_geo": L_geo.detach(),
        "L_prog": L_prog.detach(),
        "L_trait": L_trait.detach(),
        "deltaE_pred": deltaE_pred.detach(),
        "deltaP_pred": deltaP_pred.detach(),
    }
