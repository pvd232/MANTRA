
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional, Tuple, List, Dict, Union
from torch.utils.checkpoint import checkpoint

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=1048576):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        dim = torch.arange(hidden_size // 2).float().unsqueeze(0)
        div_term = torch.exp(-math.log(10000.0) * (2 * dim) / hidden_size)
        angle = pos * div_term
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(angle)
        pe[:, 1::2] = torch.cos(angle)
        self.register_buffer("pe", pe)

    def forward(self, idx):
        return self.pe[idx]

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size * 4)
        self.layer2 = nn.Linear(hidden_size * 4, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

class HybridTransformerBlock(nn.Module):
    def __init__(self, hidden_size, window_size=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.window_size = window_size
        
    def forward(self, x, memory_state=None, attn_mask=None):
        h_norm = self.norm1(x)
        T = x.shape[1]
        if attn_mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            if self.window_size is not None:
                indices = torch.arange(T, device=x.device)
                row_idx = indices.view(-1, 1)
                col_idx = indices.view(1, -1)
                window_mask = (col_idx < row_idx - self.window_size)
                mask = mask | window_mask
            attn_mask = mask
        
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, attn_mask=attn_mask)
        h = x + attn_out
        h = h + self.mlp(self.norm2(h))
        return h, memory_state, {}

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


class CentroidAddressableManifold(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        n_buckets: int = 512,
        slots_per_bucket: int = 32,
        persistent_slots: int = 31,
        tau: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_buckets = n_buckets
        self.slots_per_bucket = slots_per_bucket
        self.persistent_slots = persistent_slots
        self.rolling_slots = slots_per_bucket - persistent_slots
        self.total_slots = n_buckets * slots_per_bucket
        self.tau = tau

        # Global Memory Buffers (BSZ=1)
        self.register_buffer("slot_values", torch.zeros(1, self.total_slots, hidden_size))
        self.register_buffer("slot_keys", torch.zeros(1, self.total_slots, hidden_size))
        self.register_buffer("slot_counters", torch.zeros(1, n_buckets, dtype=torch.long))
        self.register_buffer("slot_tids", torch.full((1, self.total_slots), -1, dtype=torch.long))
        self.register_buffer("slot_regret", torch.zeros(1, self.total_slots))

        centroid_weights = torch.randn(n_buckets, hidden_size)
        centroid_weights = F.normalize(centroid_weights, dim=-1)
        self.centroid_codebook = nn.Parameter(centroid_weights)

        self.register_buffer("token_ema", torch.zeros(vocab_size))
        self.register_buffer("token_var", torch.full((vocab_size,), 0.1))

        ranks = torch.arange(vocab_size).float()
        token_weights = math.log(vocab_size + math.e) / torch.log(ranks + math.e)
        self.register_buffer("token_weights", token_weights)

        self.n_bits = int(np.log2(n_buckets))
        lsh_proj = torch.randn(hidden_size, self.n_bits) / (hidden_size**0.5)
        self.register_buffer('lsh_proj', lsh_proj)

    def init_state(self, bsz, device):
        # Persistent Global State (ignores bsz)
        if self.slot_values.shape[0] != 1:
            self.slot_values = torch.zeros(1, self.total_slots, self.hidden_size, device=device)
            self.slot_keys = torch.zeros(1, self.total_slots, self.hidden_size, device=device)
            self.slot_counters = torch.zeros(1, self.n_buckets, dtype=torch.long, device=device)
            self.slot_tids = torch.full((1, self.total_slots), -1, dtype=torch.long, device=device)
            self.slot_regret = torch.zeros(1, self.total_slots, device=device)
        else:
            self.slot_values.zero_()
            self.slot_keys.zero_()
            self.slot_counters.zero_()
            self.slot_tids.fill_(-1)
            self.slot_regret.zero_()

    @torch.no_grad()
    def read(self, query_emb, slot_values, slot_keys, tids=None, alpha=0.5):
        B, T, D = query_emb.shape
        device = query_emb.device
        lex_ids = tids.view(B, T)
        lex_anchors = self.centroid_codebook[lex_ids % self.n_buckets]
        q_norm = F.normalize(query_emb, dim=-1)
        buckets = lex_ids % self.n_buckets
        offsets = buckets * self.slots_per_bucket
        if not hasattr(self, '_slot_arange') or self._slot_arange.device != device:
             self._slot_arange = torch.arange(self.slots_per_bucket, device=device)
        indices = offsets.unsqueeze(-1) + self._slot_arange
        flat_idx = indices.view(B, -1)

        # Global Expansion
        k_exp = self.slot_keys.expand(B, -1, -1)
        v_exp = self.slot_values.expand(B, -1, -1)
        b_keys = torch.gather(k_exp, 1, flat_idx.unsqueeze(-1).expand(-1, -1, D)).view(B, T, self.slots_per_bucket, D)
        b_vals = torch.gather(v_exp, 1, flat_idx.unsqueeze(-1).expand(-1, -1, self.hidden_size)).view(B, T, self.slots_per_bucket, self.hidden_size)

        unified_query = F.normalize(alpha * q_norm + (1.0 - alpha) * lex_anchors, dim=-1)
        scores = torch.bmm(unified_query.view(B*T, 1, D), b_keys.view(B*T, self.slots_per_bucket, D).transpose(1, 2)).view(B, T, self.slots_per_bucket)
        
        if tids is not None:
            gathered_tids = torch.gather(self.slot_tids.expand(B, -1), 1, flat_idx).view(B, T, self.slots_per_bucket)
            match_mask = (tids.unsqueeze(-1) == gathered_tids).float()
            has_match = (match_mask.sum(dim=-1, keepdim=True) > 0)
            probs_hard = match_mask / (match_mask.sum(dim=-1, keepdim=True) + 1e-9)
            val_hard = torch.einsum('bts,btsw->btw', probs_hard, b_vals)
            probs_soft = torch.softmax(scores / self.tau, dim=-1)
            val_soft = torch.einsum('bts,btsw->btw', probs_soft, b_vals)
            val_out = torch.where(has_match, val_hard, val_soft)
            max_sim = scores.max(dim=-1)[0]
            max_sim = torch.where(has_match.squeeze(-1), torch.full_like(max_sim, 10.0), max_sim)
        else:
            probs_soft = torch.softmax(scores / self.tau, dim=-1)
            val_out = torch.einsum('bts,btsw->btw', probs_soft, b_vals)
            max_sim = scores.max(dim=-1)[0]
        return val_out, max_sim, {"buckets": buckets}

class NexusV2(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=4, n_buckets=512, slots_per_bucket=32, persistent_slots=31, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.micro_chunk_size = kwargs.get("micro_chunk_size", 1024)
        self.ema_alpha = kwargs.get("ema_alpha", 0.01)
        self.regret_gamma = kwargs.get("regret_gamma", 1.0)
        self.alpha_val = kwargs.get("alpha", 0.5)
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = PositionalEncoding(hidden_size)
        self.blocks = nn.ModuleList([HybridTransformerBlock(hidden_size) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight 
        self.cam = CentroidAddressableManifold(hidden_size, vocab_size, n_buckets, slots_per_bucket, persistent_slots)
        self.semantic_encoder = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.dropout = nn.Dropout(0.1)
        self.norm_mem = RMSNorm(hidden_size)
        self.mem_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mem_fuse_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, start_pos=0, **kwargs):
        bsz, seq_len = x.shape
        device = x.device
        t_emb = self.tok_emb(x)
        h = t_emb + self.pos_emb(torch.arange(start_pos, start_pos + seq_len, device=device))
        h = self.dropout(h)
        h_list = []
        for i in range(0, seq_len, 1024):
            h_c = h[:, i:i+1024]
            for block in self.blocks: h_c, _, _ = block(h_c)
            h_list.append(h_c)
        h_final = self.norm_out(torch.cat(h_list, dim=1))

        def _chunked_cam_forward(h_fin, t_e, id_e, v_low, k_low, c_low, tid_low, r_low):
            h_f_list = []; f_ret_val = []; f_q = []; f_k = []; f_masks = []
            curr_v_inner, curr_k_inner, curr_c, curr_tid, curr_r_inner = v_low, k_low, c_low, tid_low, r_low
            for t in range(0, seq_len, self.micro_chunk_size):
                h_u, t_u, id_u = h_fin[:, t:t+self.micro_chunk_size], t_e[:, t:t+self.micro_chunk_size], id_e[:, t:t+self.micro_chunk_size]
                val_hat, max_sim, diag = self.cam.read(h_u, curr_v_inner, curr_k_inner, tids=id_u, alpha=self.alpha_val)
                h_f = h_u + self.mem_proj(self.norm_mem(val_hat))
                h_f_list.append(h_f)
                logits_f = self.head(h_f) / (self.hidden_size**0.5)
                g_entropy = -(torch.softmax(logits_f, dim=-1) * torch.log_softmax(logits_f, dim=-1)).sum(dim=-1, keepdim=True).detach()
                mu, var = self.cam.token_ema[id_u].unsqueeze(-1), self.cam.token_var[id_u].unsqueeze(-1)
                eff_gate = torch.max((((g_entropy - mu) / torch.sqrt(var + 0.2)) * (1.0 - max_sim).unsqueeze(-1) > self.regret_gamma).float(), ( (start_pos + t + torch.arange(h_u.shape[1], device=device).unsqueeze(0).unsqueeze(-1)) < 64).float())
                buckets = diag["buckets"]
                indices = (buckets * self.cam.slots_per_bucket + torch.gather(curr_c, 1, buckets)).long()
                V_comb = torch.cat([t_u[..., :self.hidden_size//2], h_u[..., self.hidden_size//2:]], dim=-1)
                V_comb_norm = F.normalize(V_comb, dim=-1)
                curr_v_inner = curr_v_inner.scatter(1, indices.unsqueeze(-1).expand(-1, -1, self.hidden_size), (1.0 - eff_gate) * torch.gather(curr_v_inner, 1, indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)) + eff_gate * V_comb)
                curr_k_inner = curr_k_inner.scatter(1, indices.unsqueeze(-1).expand(-1, -1, self.hidden_size), (1.0 - eff_gate) * torch.gather(curr_k_inner, 1, indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)) + eff_gate * V_comb_norm)
                update_mask = (eff_gate.squeeze(-1) > 0.5).float()
                curr_tid = curr_tid.scatter(1, indices, torch.where(update_mask > 0, id_u, torch.gather(curr_tid, 1, indices)))
                curr_c = (curr_c.scatter_add(1, buckets, update_mask.long())) % self.cam.slots_per_bucket
                f_masks.append(eff_gate)
            return torch.cat(h_f_list, dim=1), curr_v_inner, curr_k_inner, curr_c, curr_tid, curr_r_inner, None, None, None, torch.cat(f_masks, dim=1)

        h_fused_all_final, final_v, final_k, final_c, final_tid, final_r, _, _, _, final_masks = _chunked_cam_forward(h_final, t_emb, x, self.cam.slot_values.expand(bsz, -1, -1), self.cam.slot_keys.expand(bsz, -1, -1), self.cam.slot_counters.expand(bsz, -1), self.cam.slot_tids.expand(bsz, -1), self.cam.slot_regret.expand(bsz, -1))
        
        # Global Knowledge Accumulation
        for b in range(bsz):
            self.cam.slot_values.copy_(final_v[b:b+1].detach())
            self.cam.slot_keys.copy_(final_k[b:b+1].detach())
            self.cam.slot_counters.copy_(final_c[b:b+1])
            self.cam.slot_tids.copy_(final_tid[b:b+1])
        
        h_fused_all = self.mem_fuse_norm(h_fused_all_final)
        logits = self.head(h_fused_all) / (self.hidden_size**0.5)
        return logits, h_fused_all, None, None, None, final_masks
