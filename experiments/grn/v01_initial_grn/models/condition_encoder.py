from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor


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
