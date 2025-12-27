import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm
import json
import numpy as np
import sys

# Local imports from experiment
sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "v3_functional_slotting/models"))

from v4_models import GRNGNN_V4
from adapter_v4 import NexusAdapter_V4
from mantra.nexus.tokenizer import MLFGTokenizer
from mantra.nexus.datasets import MLFGTokenDataset
from mantra.nexus.configs import NexusConfig

def train_v4():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path(__file__).parent
    
    # 1. Setup Config & Data
    npz_path = Path("/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/train.npz")
    npz_data = np.load(npz_path)
    n_regulators = int(npz_data["reg_idx"].max()) + 1
    n_programs = npz_data["deltaP_obs"].shape[1]
    
    cfg = NexusConfig(
        n_regulators=n_regulators,
        n_programs=n_programs,
        batch_size=1, # Persistent manifold still BSZ=1 focus
        max_epochs=1,
        lr=5e-5, # Joint training needs lower LR for stability
        alpha=0.8
    )
    
    tokenizer = MLFGTokenizer(n_regulators=cfg.n_regulators, n_programs=cfg.n_programs)
    cfg.vocab_size = tokenizer.vocab_size
    
    dataset = MLFGTokenDataset(npz_path, tokenizer)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    # 2. Models Initialization
    # Baseline for warm-start
    baseline_ckpt_path = "/home/machina/MANTRA/out/models/grn/hvg100/grn_k562_energy_prior.pt"
    ckpt_grn = torch.load(baseline_ckpt_path, map_location=device)
    
    grn = GRNGNN_V4(
        n_regulators=ckpt_grn["n_regulators"],
        n_genes=ckpt_grn["n_genes"],
        n_layers=ckpt_grn["grn_model_cfg"]["n_layers"],
        gene_emb_dim=ckpt_grn["grn_model_cfg"]["gene_emb_dim"],
        hidden_dim=ckpt_grn["grn_model_cfg"]["hidden_dim"],
        d_nexus=cfg.hidden_size,
        dropout=0.1,
        use_dose=False
    ).to(device)
    
    # Copy weights from baseline for the common parts
    grn.load_state_dict(ckpt_grn["model_state_dict"], strict=False)
    
    # Nexus Adapter V4
    class_map_path = "/home/machina/MANTRA/experiments/nexus/v3_functional_slotting/models/token_to_class_id.npy"
    adapter = NexusAdapter_V4(cfg, tokenizer, class_map_path=class_map_path).to(device)
    
    # Warm-start Nexus from V3 weights
    v3_ckpt_path = "/home/machina/MANTRA/experiments/nexus/v3_functional_slotting/checkpoints/model_v3_alpha08.pt"
    adapter.load_state_dict(torch.load(v3_ckpt_path, map_location=device))
    
    W = torch.from_numpy(ckpt_grn["W"]).to(device)
    A = torch.from_numpy(ckpt_grn["A"]).to(device)
    
    optimizer = optim.Adam([
        {"params": grn.parameters()},
        {"params": adapter.parameters()}
    ], lr=cfg.lr)
    
    criterion = nn.MSELoss()
    
    # 3. Joint Training Loop
    grn.train()
    adapter.train()
    adapter.model.cam.init_state(cfg.batch_size, device)
    
    print("Starting Phase 2 Recursive Joint Training...")
    
    for epoch in range(cfg.max_epochs):
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            optimizer.zero_grad()
            
            reg_idx = batch["reg_idx"].to(device)
            dose = batch["dose"].to(device)
            target_obs = batch["target"].to(device) # DeltaP [B, K]
            
            # Step A: Nexus Retrieval
            # This generates the recursive signal and the global correction
            nexus_outputs = adapter(reg_idx, dose)
            nexus_signal = nexus_outputs["nexus_signal"] # [B, D]
            deltaP_corr_global = nexus_outputs["deltaP_corr"] # [B, K]
            
            # Step B: Recursive GNN Forward
            deltaE_pred = grn(reg_idx, dose, A, nexus_signal=nexus_signal)
            deltaP_gnn = deltaE_pred @ W
            
            # Total Prediction
            deltaP_final = deltaP_gnn + deltaP_corr_global
            
            loss = criterion(deltaP_final, target_obs)
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=loss.item())
            
    # 4. Save
    out_dir = exp_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "grn_state": grn.state_dict(),
        "adapter_state": adapter.state_dict(),
        "cfg": cfg
    }, out_dir / "model_v4_recursive.pt")
    print(f"V4 Joint Model Saved to {out_dir}")

if __name__ == "__main__":
    train_v4()
