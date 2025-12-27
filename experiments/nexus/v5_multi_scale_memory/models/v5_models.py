import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import sys
from pathlib import Path

# Local Import from v3
sys.path.append(str(Path(__file__).parent.parent.parent / "v3_functional_slotting/models"))
from nexus_v3 import CentroidAddressableManifold as CAM_Base

class MultiScaleCAM(nn.Module):
    """
    Dual-Grain Memory: Fine (Gene-level) + Coarse (Pathway-level).
    """
    def __init__(self, hidden_size, vocab_size, n_buckets_fine=4096, n_buckets_coarse=512, class_map_path=None):
        super().__init__()
        # Fine-grained: Addressed by regulator ID
        self.cam_fine = CAM_Base(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            n_buckets=n_buckets_fine,
            slots_per_bucket=4,
            class_map_path=None # Use raw IDs for addressing
        )
        
        # Coarse-grained: Addressed by functional class ID
        self.cam_coarse = CAM_Base(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            n_buckets=n_buckets_coarse,
            slots_per_bucket=32,
            class_map_path=class_map_path
        )
        
        # Shared Token Statistics for Regret Gating
        self.register_buffer("token_ema", torch.zeros(vocab_size))
        self.register_buffer("token_var", torch.ones(vocab_size))
        self.ema_alpha = 0.01

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def init_state(self, bsz, device):
        self.cam_fine.init_state(bsz, device)
        self.cam_coarse.init_state(bsz, device)
        if self.token_ema.device != device:
            self.token_ema = self.token_ema.to(device)
            self.token_var = self.token_var.to(device)

    def read(self, query, tids, alpha=None):
        bsz = query.shape[0]
        # 1. Query Fine: Lexical Focus (alpha=0.5) 
        # Focus on "Rare Recall" for specific regulator outliers
        v_fine, sim_fine, diag_fine = self.cam_fine.read(
            query, 
            self.cam_fine.slot_values.expand(bsz, -1, -1),
            self.cam_fine.slot_keys.expand(bsz, -1, -1),
            tids=tids, 
            alpha=0.5
        )
        # 2. Query Coarse: Pure Semantic (alpha=1.0)
        # Focus on "Common Recall" for pathway-level generalizations
        v_coarse, sim_coarse, diag_coarse = self.cam_coarse.read(
            query, 
            self.cam_coarse.slot_values.expand(bsz, -1, -1),
            self.cam_coarse.slot_keys.expand(bsz, -1, -1),
            tids=tids, # Pass tids for bucket lookup
            alpha=1.0  # Decoupling handled by alpha-gate in NexusV3
        )
        
        # Gated Fusion
        fusion_gate = self.gate(torch.cat([v_fine.detach(), v_coarse.detach()], dim=-1))
        v_out = (1.0 - fusion_gate) * v_coarse + fusion_gate * v_fine
        
        return v_out, (sim_fine + sim_coarse)/2, {
            "gate": fusion_gate, 
            "diag_fine": diag_fine, 
            "diag_coarse": diag_coarse,
            "sim_fine": sim_fine,
            "sim_coarse": sim_coarse
        }

    def update_step(self, query, tids, h_fused, logits, diag, regret_gamma=1.5):
        bsz, seq_len, D = query.shape
        device = query.device
        
        # 1. Calculate Surprisal (Entropy)
        # Higher entropy = more "surprised" = more likely to write
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1, keepdim=True).detach()
        
        # 2. Global Stats Update (EMA)
        with torch.no_grad():
            flat_ids = tids.view(-1)
            flat_ent = entropy.view(-1)
            # Simple EMA update for token-specific surprisal
            # In a production run, we'd use scatter_add for exactness, but this is a diagnostic experiment
            self.token_ema[flat_ids] = (1 - self.ema_alpha) * self.token_ema[flat_ids] + self.ema_alpha * flat_ent
            self.token_var[flat_ids] = (1 - self.ema_alpha) * self.token_var[flat_ids] + self.ema_alpha * (flat_ent - self.token_ema[flat_ids])**2

        # 3. Z-Scored Regret Gating
        mu = self.token_ema[tids].unsqueeze(-1)
        var = self.token_var[tids].unsqueeze(-1)
        z_score = (entropy - mu) / torch.sqrt(var + 1e-6)
        
        # Decoupled Budgets (V5.1 Relaxed Bias):
        # Fine (Gene) is rigid: only high-surprisal outliers (Z > 2.0)
        # Coarse (Pathway) is permissive: 100% density to establish prior
        
        gate_fine = (z_score * (1.0 - diag["sim_fine"].unsqueeze(-1)) > regret_gamma * 2).float()
        gate_coarse = torch.ones_like(z_score) # V5.1 Salvage: 100% Coarse Density
        
        def _update_manifold(manifold, q, tid, h_f, d, eff_gate):
            # Only update where eff_gate > 0
            buckets = d["buckets"]
            counters = manifold.slot_counters.expand(bsz, -1)
            indices = (buckets * manifold.slots_per_bucket + torch.gather(counters, 1, buckets)).long()
            
            V_comb = h_f 
            V_comb_norm = F.normalize(V_comb, dim=-1)
            
            # Apply Gated Scatter
            # (Note: we only update the specific slots where eff_gate is 1)
            mask = eff_gate.squeeze(-1) > 0.5
            if not mask.any(): return
            
            # Simplified update pass
            manifold.slot_values.data = manifold.slot_values.data.scatter(1, indices.unsqueeze(-1).expand(-1, -1, D), (1.0 - eff_gate) * torch.gather(manifold.slot_values.data, 1, indices.unsqueeze(-1).expand(-1, -1, D)) + eff_gate * V_comb)
            manifold.slot_keys.data = manifold.slot_keys.data.scatter(1, indices.unsqueeze(-1).expand(-1, -1, D), (1.0 - eff_gate) * torch.gather(manifold.slot_keys.data, 1, indices.unsqueeze(-1).expand(-1, -1, D)) + eff_gate * V_comb_norm)
            manifold.slot_tids.data = manifold.slot_tids.data.scatter(1, indices, torch.where(mask, tid, torch.gather(manifold.slot_tids.data, 1, indices)))
            manifold.slot_counters.data = (manifold.slot_counters.data.scatter_add(1, buckets, mask.long())) % manifold.slots_per_bucket

        _update_manifold(self.cam_fine, query, tids, h_fused, diag["diag_fine"], gate_fine)
        _update_manifold(self.cam_coarse, query, tids, h_fused, diag["diag_coarse"], gate_coarse)
        
        return {"gate_fine": gate_fine.mean().item(), "gate_coarse": gate_coarse.mean().item()}

class NexusV5(nn.Module):
    """
    Nexus wrapper for MultiScaleCAM.
    Manages dual manifolds (Gene + Pathway) with gated fusion.
    """
    def __init__(self, vocab_size, hidden_size, num_layers=4, n_buckets_fine=4096, n_buckets_coarse=512, class_map_path=None, **kwargs):
        super().__init__()
        from nexus_v3 import PositionalEncoding, HybridTransformerBlock, RMSNorm
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.alpha_val = kwargs.get("alpha", 0.8)
        self.micro_chunk_size = kwargs.get("micro_chunk_size", 1024)
        
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = PositionalEncoding(hidden_size)
        self.blocks = nn.ModuleList([HybridTransformerBlock(hidden_size) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight 
        
        self.cam = MultiScaleCAM(hidden_size, vocab_size, n_buckets_fine, n_buckets_coarse, class_map_path)
        
        self.norm_mem = RMSNorm(hidden_size)
        self.mem_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mem_fuse_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, start_pos=0, **kwargs):
        bsz, seq_len = x.shape
        device = x.device
        h = self.tok_emb(x) + self.pos_emb(torch.arange(start_pos, start_pos + seq_len, device=device))
        h = self.dropout(h)
        
        # Transformer pass
        for block in self.blocks: h_block, _, _ = block(h)
        h_final = self.norm_out(h_block)

        # Multi-Scale CAM retrieval
        val_hat, max_sim, diag = self.cam.read(h_final, x, alpha=self.alpha_val)
        h_fused = h_final + self.mem_proj(self.norm_mem(val_hat))
        h_fused = self.mem_fuse_norm(h_fused)
        
        logits = self.head(h_fused) / (self.hidden_size**0.5)

        # Dual-Manifold Update
        if self.training:
            self.cam.update_step(
                h_final, x, h_fused, logits,
                diag
            )
        
        return logits, h_fused, None, None, None, None
