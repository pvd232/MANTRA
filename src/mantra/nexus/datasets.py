import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional
from mantra.nexus.tokenizer import MLFGTokenizer

class MLFGTokenDataset(Dataset):
    """
    Dataset that lazily tokenizes NPZ records.
    """
    def __init__(
        self, 
        npz_path: Path, 
        tokenizer: MLFGTokenizer, 
        top_p: int = 16,
        max_seq_len: int = 32
    ):
        self.npz_path = Path(npz_path)
        data = np.load(self.npz_path, allow_pickle=False)
        
        self.reg_idx = data["reg_idx"].astype(np.int64)
        self.deltaP_obs = data["deltaP_obs"].astype(np.float32)
        self.dose = data.get("dose", np.zeros_like(self.reg_idx)).astype(np.float32)
        
        self.tokenizer = tokenizer
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.n_samples = self.reg_idx.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        reg = self.reg_idx[idx]
        dose = self.dose[idx]
        if dose.ndim > 0: dose = dose[0]
        
        deltaP = torch.from_numpy(self.deltaP_obs[idx])
        
        tokens = self.tokenizer.encode_record(
            reg_idx=reg,
            dose_val=dose,
            deltaP=deltaP,
            top_p=self.top_p
        )
        
        # Padding
        padded = torch.full((self.max_seq_len,), self.tokenizer.pad_token, dtype=torch.long)
        l = min(len(tokens), self.max_seq_len)
        padded[:l] = tokens[:l]
        
        # Trait Delta Readout (Optional)
        deltaY = None
        if hasattr(self, "deltaY_obs"):
            deltaY = torch.from_numpy(self.deltaY_obs[idx]).float()
        
        res = {
            "input_ids": padded,
            "target": deltaP, # DeltaP
            "reg_idx": torch.tensor(reg, dtype=torch.long),
            "dose": torch.tensor(dose, dtype=torch.float32)
        }
        if deltaY is not None:
            res["target_y"] = deltaY
            
        return res
        
