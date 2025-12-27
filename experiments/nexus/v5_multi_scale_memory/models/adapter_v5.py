import torch
import torch.nn as nn
from typing import Optional, Dict
import sys
from pathlib import Path

# Local imports
sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent.parent.parent / "v3_functional_slotting/models"))
from v5_models import NexusV5
from mantra.nexus.tokenizer import MLFGTokenizer
from mantra.nexus.configs import NexusConfig

class NexusAdapter_V5(nn.Module):
    """
    Nexus V5 Adapter: Supports Hierarchical (Multi-Scale) Memory.
    """
    def __init__(self, cfg, tokenizer, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        
        self.model = NexusV5(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_layers=4,
            n_buckets_fine=4096,
            n_buckets_coarse=512,
            class_map_path=kwargs.get("class_map_path"),
            alpha=cfg.alpha
        )
        
        self.proj = nn.Linear(cfg.hidden_size, cfg.n_programs)
        # SMR Bridge: Map programs to traits (MCH, RDW, IRF)
        self.trait_proj = nn.Linear(cfg.n_programs, 3) 

    def forward(self, reg_idx, dose, state_bin=0):
        # Similar to V4 but with V5 model
        batch_tokens = []
        for i in range(reg_idx.shape[0]):
            r = reg_idx[i].item()
            d = dose[i].item() if dose is not None else 0.0
            t = [
                self.tokenizer.reg_offset + r,
                self.tokenizer.dose_offset + min(int(d * self.tokenizer.n_dose_bins), self.tokenizer.n_dose_bins - 1),
                self.tokenizer.state_offset + state_bin
            ]
            batch_tokens.append(torch.tensor(t))
        x = torch.stack(batch_tokens).to(reg_idx.device)
        
        # Nexus V5 Forward
        logits, h_fused, _, _, _, masks = self.model(x)
        h_query = h_fused[:, 2, :] # [B, D]
        
        # Correction
        deltaP_corr = self.proj(h_query)
        
        # Trait Prediction (SMR Bridge)
        deltaY_pred = self.trait_proj(deltaP_corr)
        
        return {
            "nexus_signal": h_query,
            "deltaP_corr": deltaP_corr,
            "deltaY_pred": deltaY_pred,
            "masks": masks
        }
