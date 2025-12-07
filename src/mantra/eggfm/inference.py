# src/mantra/eggfm/inference.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn, Tensor

from mantra.eggfm.models import EnergyMLP


class EnergyScorer:
    """
    Wraps a trained EnergyMLP + normalization (+ optional projection)
    so we can compute energies in a consistent way.

    Supports:
      - HVG / gene space: x -> normalize -> E(x)
      - Embedding space: x -> project (PCA) -> normalize -> E_z(z)
    """

    def __init__(
        self,
        energy_model: nn.Module,
        mean: Optional[Tensor],
        std: Optional[Tensor],
        var_names: Optional[Sequence[str]] = None,
        proj_matrix: Optional[Tensor] = None,   # [G, d] for embedding case
        space: str = "hvg",
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.energy_model = energy_model.to(self.device)
        self.energy_model.eval()
        for p in self.energy_model.parameters():
            p.requires_grad_(False)

        self.space = space

        # mean/std are always in the *model feature space*: [D_model]
        self.mean = None if mean is None else mean.to(self.device).view(1, -1)
        self.std = None if std is None else std.to(self.device).view(1, -1)

        # gene feature metadata (for HVG space alignment; optional)
        self.var_names: Optional[List[str]] = None
        self._name_to_idx: Optional[Dict[str, int]] = None
        if var_names is not None:
            self.var_names = [str(v) for v in var_names]
            self._name_to_idx = {name: i for i, name in enumerate(self.var_names)}

        # optional projection (for embedding case): [G_raw, D_model]
        self.proj_matrix: Optional[Tensor] = None
        if proj_matrix is not None:
            proj_matrix = proj_matrix.to(self.device)
            self.proj_matrix = proj_matrix

    # ------------------------------------------------------------------
    # Construction helper
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> "EnergyScorer":
        """
        Load an EnergyScorer from a .pt checkpoint.

        Expects ckpt to contain something like:

            {
                "state_dict": ...,
                "model_cfg": {"hidden_dims": [...]},
                "n_genes": int,       # D_model
                "space": "hvg" or "embedding",
                "var_names": [...],   # optional, for HVG space alignment
                "mean": ...,
                "std": ...,
                # optional for embedding:
                "proj_matrix": np.ndarray [G_raw, D_model],
            }
        """
        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device or "cpu")

        n_genes = ckpt.get("n_genes")
        model_cfg = ckpt.get("model_cfg", {})
        space = ckpt.get("space", "hvg")

        # reconstruct EnergyMLP in model feature space
        energy_model = EnergyMLP(
            n_genes=n_genes,
            **model_cfg,
        )
        energy_model.load_state_dict(ckpt["state_dict"])

        def _to_tensor_or_none(key: str) -> Optional[Tensor]:
            if key not in ckpt or ckpt[key] is None:
                return None
            arr = ckpt[key]
            if isinstance(arr, Tensor):
                return arr
            return torch.as_tensor(arr, dtype=torch.float32)

        mean = _to_tensor_or_none("mean")
        std = _to_tensor_or_none("std")
        var_names = ckpt.get("var_names", None)

        proj_matrix = _to_tensor_or_none("proj_matrix")  # for embedding space

        return cls(
            energy_model=energy_model,
            mean=mean,
            std=std,
            var_names=var_names,
            proj_matrix=proj_matrix,
            space=space,
            device=device,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_tensor(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        if isinstance(x, Tensor):
            return x.to(self.device, dtype=torch.float32)
        else:
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _reorder_by_genes(
        self,
        x: Tensor,                     # [B, G_in]
        gene_names: Optional[Sequence[str]],
    ) -> Tensor:
        """
        If var_names and gene_names are provided, reorder x to match training order.
        Otherwise, assume x is already aligned.
        """
        if self.var_names is None or gene_names is None:
            return x

        if len(self.var_names) != x.shape[1]:
            raise ValueError(
                f"EnergyScorer: mismatch between model gene dim ({len(self.var_names)}) "
                f"and input x.shape[1] ({x.shape[1]})."
            )

        input_name_to_idx = {str(name): i for i, name in enumerate(gene_names)}

        try:
            indices = [input_name_to_idx[name] for name in self.var_names]
        except KeyError as e:
            missing = str(e.args[0])
            raise KeyError(
                f"EnergyScorer: gene {missing!r} from training var_names "
                f"not found in provided gene_names."
            )

        idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
        return x[:, idx]

    def _apply_normalization(self, z: Tensor) -> Tensor:
        if self.mean is None or self.std is None:
            return z
        return (z - self.mean) / self.std

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(
        self,
        x_raw: Union[Tensor, np.ndarray],
        gene_names: Optional[Sequence[str]] = None,
    ) -> Tensor:
        """
        Compute energy for a batch of states x_raw in *gene space*:

            - If space == "hvg": x_raw is already in model feature space (after alignment).
            - If space == "embedding": x_raw is in gene space; we project with proj_matrix.

        Returns energies: [B].
        """
        x = self._ensure_tensor(x_raw)  # [B, G_in]
        x = self._reorder_by_genes(x, gene_names)  # optionally align to var_names

        # If we have a projection matrix, go to embedding space
        if self.proj_matrix is not None:
            z = x @ self.proj_matrix    # [B, D_model]
        else:
            z = x                       # [B, D_model]

        z_norm = self._apply_normalization(z)      # [B, D_model]
        energy = self.energy_model(z_norm)         # [B] or [B,1]
        if energy.ndim == 2:
            energy = energy.squeeze(-1)
        return energy

    @torch.no_grad()
    def score_delta(
        self,
        x_ref: Union[Tensor, np.ndarray],
        deltaE_pred: Union[Tensor, np.ndarray],
        gene_names: Optional[Sequence[str]] = None,
    ) -> Tensor:
        """
        Convenience: score energy of x_hat = x_ref + ΔE_pred.

        x_ref: [G] or [1,G]
        deltaE_pred: [B,G]
        """
        x_ref_t = self._ensure_tensor(x_ref)
        if x_ref_t.ndim == 1:
            x_ref_t = x_ref_t.unsqueeze(0)  # [1,G]
        delta_t = self._ensure_tensor(deltaE_pred)  # [B,G]

        if x_ref_t.shape[1] != delta_t.shape[1]:
            raise ValueError(
                f"x_ref dim {x_ref_t.shape[1]} != deltaE_pred dim {delta_t.shape[1]}"
            )

        x_hat = x_ref_t + delta_t
        return self.score(x_hat, gene_names=gene_names)
