import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import sys
from pathlib import Path

# Local import
sys.path.append(str(Path(__file__).parent.parent.parent / "v3_functional_slotting/models"))
from nexus_v3 import NexusV3
from mantra.nexus.configs import NexusConfig
from mantra.nexus.tokenizer import MLFGTokenizer

class NexusAdapter_V4(nn.Module):
    """
    Nexus V4 Adapter: Provides recursive signals and global residuals.
    """
    def __init__(self, cfg: NexusConfig, tokenizer: MLFGTokenizer, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        
        self.model = NexusV3(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            n_buckets=cfg.n_buckets,
            slots_per_bucket=cfg.slots_per_bucket,
            persistent_slots=cfg.persistent_slots,
            ema_alpha=cfg.ema_alpha,
            regret_gamma=cfg.regret_gamma,
            alpha=cfg.alpha,
            class_map_path=kwargs.get("class_map_path")
        )
        
        # Output head for global residual correction
        self.proj = nn.Linear(cfg.hidden_size, cfg.n_programs)

    def forward(
        self, 
        reg_idx: torch.Tensor, 
        dose: Optional[torch.Tensor] = None, 
        state_bin: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Produce:
        1. nexus_signal: [B, D] (for recursive GNN injection)
        2. deltaP_corr: [B, K] (original global residual)
        """
        # Build header tokens [REG, DOSE, STATE]
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
        
        # Nexus Forward (Retrieval + Persistent Update)
        out = self.model(x)
        h_fused = out[1] # [B, L, D]
        
        # Extract signal from STATE token (index 2)
        h_query = h_fused[:, 2, :] # [B, D]
        
        # Compute correction
        deltaP_corr = self.proj(h_query)
        
        return {
            "nexus_signal": h_query,
            "deltaP_corr": deltaP_corr
        }
