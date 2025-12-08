# AnnDataPyTorch.py

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse


class AnnDataExpressionDataset(Dataset):
    """
    Wraps an AnnData object's X matrix (after prep()) as a PyTorch dataset.
    Uses HVG, log-normalized expression directly.
    """

    def __init__(self, X, float_dtype=np.float32):
        if sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=float_dtype)

        mean = X.mean(axis=0, keepdims=True)
        std  = X.std(axis=0, keepdims=True)
        
        # prevent divide-by-zero or tiny variance explosions
        std = np.clip(std, 1e-2, None)

        # store for later (without the extra batch dim)
        self.mean = mean.astype(float_dtype).squeeze(0)  # shape [D]
        self.std  = std.astype(float_dtype).squeeze(0)   # shape [D]
        
        
        self.X = (X - mean) / std



    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.X[idx])
