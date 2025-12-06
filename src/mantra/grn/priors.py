# src/mantra/grn/priors.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import nn, Tensor

from mantra.eggfm.inference import EnergyScorer


class EnergyScorerPrior(nn.Module):
    """
    Wraps an EnergyScorer as a frozen prior: x_hat -> energy.

    GRN does not care whether the underlying energy lives in HVG
    space or an embedding; that logic is inside EnergyScorer.
    """

    def __init__(
        self,
        scorer: EnergyScorer,
        gene_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.scorer = scorer
        # optional canonical gene order for GRN's feature space
        self.gene_names = list(gene_names) if gene_names is not None else None

    def forward(self, x_hat: Tensor) -> Tensor:
        # x_hat: [B, G_raw] in GRN's gene space
        return self.scorer.score(x_hat, gene_names=self.gene_names)


def build_energy_prior_from_ckpt(
    ckpt_path: str | Path,
    gene_names: Optional[Sequence[str]],
    device: Optional[torch.device] = None,
) -> EnergyScorerPrior:
    """
    Build an EnergyScorerPrior from a pre-trained EGGFM checkpoint.

    The checkpoint itself encodes:
      - HVG vs embedding space
      - normalization
      - (for embedding) projection matrix.
    """
    scorer = EnergyScorer.from_checkpoint(ckpt_path, device=device)
    prior = EnergyScorerPrior(scorer=scorer, gene_names=gene_names)
    prior.eval()
    return prior
