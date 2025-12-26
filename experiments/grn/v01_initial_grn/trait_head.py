from __future__ import annotations

from torch import nn, Tensor


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
