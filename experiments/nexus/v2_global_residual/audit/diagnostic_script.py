import torch
import numpy as np
import json
from pathlib import Path
from mantra.nexus.adapter import NexusAdapter
from mantra.nexus.tokenizer import MLFGTokenizer
from mantra.nexus.configs import NexusConfig
from mantra.grn.dataset import K562RegDeltaDataset
from torch.utils.data import DataLoader

def diagnose():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup
    with open("/home/machina/MANTRA/out/nexus/vocab.json", "r") as f:
        meta = json.load(f)
    
    cfg = NexusConfig(
        vocab_size=meta["vocab_size"],
        n_regulators=meta["n_regulators"],
        n_programs=meta["n_programs"]
    )
    tokenizer = MLFGTokenizer(n_regulators=cfg.n_regulators, n_programs=cfg.n_programs)
    nexus = NexusAdapter(cfg, tokenizer).to(device)
    nexus.model.cam.init_state(8, device) 
    nexus.load_state_dict(torch.load("/home/machina/MANTRA/out/nexus/model.pt", map_location=device))
    nexus.eval()
    
    dataset = K562RegDeltaDataset(Path("/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/val.npz"))
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 2. Metrics containers
    write_masks = []
    max_sims = []
    all_losses = []
    
    print("Running diagnostics...", flush=True)
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # Header tokens for inference
            batch_tokens = []
            for j in range(batch["reg_idx"].shape[0]):
                r = batch["reg_idx"][j].item()
                d = batch["dose"][j].item() if "dose" in batch else 0.0
                t = [
                    tokenizer.reg_offset + r,
                    tokenizer.dose_offset + min(int(d * tokenizer.n_dose_bins), tokenizer.n_dose_bins - 1),
                    tokenizer.state_offset + 0
                ]
                batch_tokens.append(torch.tensor(t))
            
            x = torch.stack(batch_tokens).to(device)
            
            # Forward call returns (logits, h_fused, final_val, final_q, final_k_sem, final_masks)
            # Oops, I need to check the return values again.
            res = nexus.model(x)
            logits, h_fused, final_val, final_q, final_k_sem, final_masks = res
            
            write_masks.append(final_masks.cpu())
            
            # Re-read to get similarity if not explicitly returned or check final_val
            # Actually, return_retrieved was true, so final_val/final_q/final_k_sem are there.
            
            if i > 100: break # Small sample
            
    # 3. Analyze write density
    all_masks = torch.cat(write_masks, dim=0) # [B, L, 1]
    avg_write_density = all_masks.mean().item()
    
    print(f"--- Nexus Diagnostic Report ---")
    print(f"Write Density: {avg_write_density:.4%}")
    
    if avg_write_density < 0.01:
        print("ALERT: Write density is extremely low. Surprisal gate might be too restrictive.")
    
    # 4. Check bucket collisions
    counters = nexus.model.cam.slot_counters
    occ = (counters > 0).float().mean().item()
    print(f"Bucket Occupancy: {occ:.4%}")
    
    full_buckets = (counters >= cfg.slots_per_bucket).float().mean().item()
    print(f"Full Buckets: {full_buckets:.4%}")

if __name__ == "__main__":
    diagnose()
