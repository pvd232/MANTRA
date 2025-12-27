import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm
import json
import numpy as np

# Local imports from experiment
sys.path.append(str(Path(__file__).parent / "models"))
from adapter_v3 import NexusAdapter
from mantra.nexus.tokenizer import MLFGTokenizer
from mantra.nexus.datasets import MLFGTokenDataset
from mantra.nexus.configs import NexusConfig
from mantra.grn.models import GRNGNN

def train():
    # 1. Setup Config & Data
    npz_path = Path("/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/train.npz")
    
    # Load metadata to get vocab sizes
    # Inferred from the NPZ
    npz_data = np.load(npz_path)
    n_regulators = int(npz_data["reg_idx"].max()) + 1
    n_programs = npz_data["deltaP_obs"].shape[1]
    
    cfg = NexusConfig(
        n_regulators=n_regulators,
        n_programs=n_programs,
        batch_size=1,
        max_epochs=1,
        lr=1e-4,
        alpha=0.8, # [V3] High semantic reliance for shared buckets
        n_buckets=4096, # [V3.1] 1-to-1 Functional Class Resolution
        slots_per_bucket=4 # [V3.1] Shallow slots to focus on semantic nuance
    )
    
    tokenizer = MLFGTokenizer(
        n_regulators=cfg.n_regulators,
        n_programs=cfg.n_programs
    )
    cfg.vocab_size = tokenizer.vocab_size
    
    dataset = MLFGTokenDataset(npz_path, tokenizer, top_p=cfg.top_p, max_seq_len=cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    # 2. Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_map_path = str(Path(__file__).parent / "models/token_to_class_id.npy")
    adapter = NexusAdapter(cfg, tokenizer, class_map_path=class_map_path).to(device)
    
    # [Baseline Setup for Residual Training]
    baseline_ckpt_path = "/home/machina/MANTRA/out/models/grn/hvg100/grn_k562_energy_prior.pt"
    ckpt = torch.load(baseline_ckpt_path, map_location=device)
    
    # Filter config for GRNGNN
    grn_cfg = {k: v for k, v in ckpt["grn_model_cfg"].items() if k in ["n_layers", "gene_emb_dim", "hidden_dim", "dropout", "use_dose"]}
    
    baseline_model = GRNGNN(
        n_regulators=ckpt["n_regulators"],
        n_genes=ckpt["n_genes"],
        **grn_cfg
    ).to(device)
    baseline_model.load_state_dict(ckpt["model_state_dict"])
    baseline_model.eval()
    
    W = torch.from_numpy(ckpt["W"]).to(device) # [G, K]
    A = torch.from_numpy(ckpt["A"]).to(device) # [G, G]
    
    optimizer = optim.Adam(adapter.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    adapter.train()
    # Initialize state once for persistence
    # We use a batch size of cfg.batch_size for the persistent buffers
    adapter.model.cam.init_state(cfg.batch_size, device)
    
    for epoch in range(cfg.max_epochs):
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        total_loss = 0
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device) # [B, L]
            target_obs = batch["target"].to(device) # [B, K]
            reg_idx = batch["reg_idx"].to(device)
            dose = batch["dose"].to(device)
            
            # Compute baseline residual
            with torch.no_grad():
                deltaE_base = baseline_model(reg_idx=reg_idx, dose=dose, A=A)
                deltaP_base = deltaE_base @ W
                target_resid = target_obs - deltaP_base
            
            # Forward
            # Note: Model internal state persists now because we removed auto-init
            out = adapter.model(input_ids)
            h_fused = out[1] # [B, L, D]
            
            # [CRITICAL FIX] Predict from the STATE token (index 2)
            # This ensures it's a prediction based on REG+DOSE+STATE 
            # and potentially previous memory context, 
            # but NOT the currently-being-written content.
            h_query = h_fused[:, 2, :] # [B, D]
            
            pred_resid = adapter.proj(h_query)
            loss = criterion(pred_resid, target_resid)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch} complete. Avg Loss: {total_loss / len(loader)}")

    # 4. Save
    output_dir = Path(__file__).parent / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), output_dir / "model_v3_1_hires.pt")
    tokenizer.save_vocab(output_dir / "vocab.json")
    print(f"Saved model to {output_dir / 'model_v3_1_hires.pt'}")

import numpy as np
if __name__ == "__main__":
    train()
