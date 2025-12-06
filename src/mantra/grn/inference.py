# src/mantra/grn/inference.py

from __future__ import annotations

from typing import Optional, Dict

import torch
from torch import nn, Tensor

from mantra.grn.models import GRNGNN, TraitHead

class GRNInference:
    """
    Inference wrapper for a trained GRN + energy prior.

    Given:
      - grn_model
      - optional trait_head
      - A (gene graph), x_ref, W (cNMF loadings)
      - energy_fn (HVG or embedding prior)

    Provides:
      - predict_batch(batch): uses same batch dict interface as training
      - predict(reg_idx, dose=None): minimalist convenience for (r,d) → ΔE, ΔP, Δy, energy
    """
    def __init__(
        self,
        grn_model: GRNGNN,
        A: Tensor,
        x_ref: Tensor,
        W: Tensor,
        energy_fn: nn.Module,
        trait_head: Optional[TraitHead] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.grn_model = grn_model.to(self.device).eval()
        self.trait_head = trait_head.to(self.device).eval() if trait_head is not None else None

        self.A = A.to(self.device)
        self.x_ref = x_ref.to(self.device)
        self.W = W.to(self.device)
        self.energy_fn = energy_fn.to(self.device).eval()

        for p in self.grn_model.parameters():
            p.requires_grad_(False)
        if self.trait_head is not None:
            for p in self.trait_head.parameters():
                p.requires_grad_(False)
        for p in self.energy_fn.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def predict_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        batch keys:
            reg_idx: [B]
            dose:    [B] or None
        """
        reg_idx = batch["reg_idx"].to(self.device)
        dose = batch.get("dose", None)
        if dose is not None:
            dose = dose.to(self.device)

        deltaE_pred = self.grn_model(reg_idx=reg_idx, dose=dose, A=self.A)  # [B, G]
        x_hat = self.x_ref.unsqueeze(0) + deltaE_pred                       # [B, G]
        energy = self.energy_fn(x_hat)                                      # [B]

        deltaP_pred = deltaE_pred @ self.W                                  # [B, K]

        out: Dict[str, Tensor] = {
            "deltaE_pred": deltaE_pred.cpu(),
            "deltaP_pred": deltaP_pred.cpu(),
            "energy": energy.cpu(),
        }

        if self.trait_head is not None:
            deltaY_pred = self.trait_head(deltaP_pred)                      # [B, T]
            out["deltaY_pred"] = deltaY_pred.cpu()

        return out

    @torch.no_grad()
    def predict(
        self,
        reg_idx: Tensor,      # [B] or scalar long
        dose: Optional[Tensor] = None,  # [B] or scalar float, optional
    ) -> Dict[str, Tensor]:
        """
        Convenience wrapper around predict_batch.
        """
        if reg_idx.dim() == 0:
            reg_idx = reg_idx.view(1)
        batch = {"reg_idx": reg_idx}

        if dose is not None:
            if dose.dim() == 0:
                dose = dose.view(1)
            batch["dose"] = dose

        return self.predict_batch(batch)
