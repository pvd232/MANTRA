import torch
import torch.nn as nn
from typing import Optional, Tuple
from mantra.nexus.hybrid_transformer_v68b import HybridTransformerV68b
from mantra.nexus.configs import NexusConfig
from mantra.nexus.tokenizer import MLFGTokenizer

class NexusAdapter(nn.Module):
    """
    Adapter that integrates Nexus memory into MANTRA.
    """
    def __init__(self, cfg: NexusConfig, tokenizer: MLFGTokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        
        self.model = HybridTransformerV68b(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            n_buckets=cfg.n_buckets,
            slots_per_bucket=cfg.slots_per_bucket,
            persistent_slots=cfg.persistent_slots,
            ema_alpha=cfg.ema_alpha,
            regret_gamma=cfg.regret_gamma,
            alpha=cfg.alpha
        )
        
        # Output head for Mode B: supervised residual projection
        # Map transformer hidden state back to program space
        self.proj = nn.Linear(cfg.hidden_size, cfg.n_programs)

    def forward(
        self, 
        reg_idx: torch.Tensor, 
        dose: torch.Tensor, 
        deltaP_baseline: Optional[torch.Tensor] = None,
        state_bin: int = 0
    ) -> torch.Tensor:
        """
        Produce a correction Δa_corr in program space.
        """
        # 1. (Optionally) build or retrieve record tokens for the batch
        # For simplicity in the adapter, we might assume header-only first or
        # pass pre-baked tokens if we're in training.
        # But for inference, we use the header tokens.
        
        # TODO: Refine how many tokens to pass. If we only have reg/dose, 
        # sequence is short. If we have partial deltaP_baseline, we can use it.
        
        # Let's assume inference uses [REG, DOSE, STATE] tokens
        batch_tokens = []
        for i in range(reg_idx.shape[0]):
            r = reg_idx[i].item()
            d = dose[i].item() if dose is not None else 0.0
            
            # Header only
            t = [
                self.tokenizer.reg_offset + r,
                self.tokenizer.dose_offset + min(int(d * self.tokenizer.n_dose_bins), self.tokenizer.n_dose_bins - 1),
                self.tokenizer.state_offset + state_bin
            ]
            batch_tokens.append(torch.tensor(t))
            
        x = torch.stack(batch_tokens).to(reg_idx.device)
        
        # 2. Nexus forward (memory read/write)
        # HybridTransformerV68b.forward returns (logits, retrieved_values, ...)
        # We want the output hidden state or logits. 
        # Let's check V68b implementation for what it returns.
        out = self.model(x)
        # HybridTransformerV68b returns (logits, h_fused, final_val, ...)
        h_fused = out[1] # [B, L, D]
        
        # Project last token hidden state to program space
        h_last = h_fused[:, -1, :] # [B, D]
        
        deltaP_corr = self.proj(h_last)
        
        return deltaP_corr
