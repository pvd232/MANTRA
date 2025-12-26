# src/mantra/grn/dataset.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class K562RegDeltaDataset(Dataset):
    """
    NPZ-backed dataset for GRN training.

    Expected keys in the .npz:
        - reg_idx:    [N]         int64, regulator index per sample
        - deltaE:     [N, G]      float32, gene-level ΔE_obs
        - deltaP_obs: [N, K]      float32 (optional)
        - deltaY_obs: [N, T]      float32 (optional)
        - dose:       [N] or [N,1] float32 (optional)

    Notes:
        - n_genes inferred from deltaE.shape[1]
        - n_regulators inferred as max(reg_idx) + 1
    """

    def __init__(self, npz_path: Path) -> None:
        npz_path = Path(npz_path)
        data = np.load(npz_path, allow_pickle=False)

        self.reg_idx = data["reg_idx"].astype(np.int64)
        self.deltaE = data["deltaE"].astype(np.float32)

        self.deltaP_obs = (
            data["deltaP_obs"].astype(np.float32)
            if "deltaP_obs" in data.files
            else None
        )
        self.deltaY_obs = (
            data["deltaY_obs"].astype(np.float32)
            if "deltaY_obs" in data.files
            else None
        )
        self.dose = (
            data["dose"].astype(np.float32)
            if "dose" in data.files
            else None
        )

        self.n_samples = self.reg_idx.shape[0]
        self.n_genes = self.deltaE.shape[1]
        self.n_regulators = int(self.reg_idx.max()) + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {
            "reg_idx": torch.as_tensor(self.reg_idx[idx], dtype=torch.long),
            "deltaE": torch.from_numpy(self.deltaE[idx]),  # [G]
        }
        if self.deltaP_obs is not None:
            batch["deltaP_obs"] = torch.from_numpy(self.deltaP_obs[idx])
        if self.deltaY_obs is not None:
            batch["deltaY_obs"] = torch.from_numpy(self.deltaY_obs[idx])
        if self.dose is not None:
            batch["dose"] = torch.as_tensor(self.dose[idx], dtype=torch.float32)
        return batch
