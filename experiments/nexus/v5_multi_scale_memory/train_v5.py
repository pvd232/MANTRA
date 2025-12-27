import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm
import json
import numpy as np
import sys

# Local imports
sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "v4_recursive_injection/models"))
sys.path.append(str(Path(__file__).parent.parent / "v3_functional_slotting/models"))

from v5_models import MultiScaleCAM, NexusV5
from v5_gnn import GRNGNN_V5
from adapter_v5 import NexusAdapter_V5
from mantra.nexus.tokenizer import MLFGTokenizer
from mantra.nexus.datasets import MLFGTokenDataset
from mantra.nexus.configs import NexusConfig

def train_v5():
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
        batch_size=1,
        max_epochs=1,
        lr=3e-5, # Multi-scale needs even lower LR for stable fusion
        alpha=0.8
    )
    
    tokenizer = MLFGTokenizer(n_regulators=cfg.n_regulators, n_programs=cfg.n_programs)
    cfg.vocab_size = tokenizer.vocab_size
    
    dataset = MLFGTokenDataset(npz_path, tokenizer)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    # 2. Models Initialization
    baseline_ckpt_path = "/home/machina/MANTRA/out/models/grn/hvg100/grn_k562_energy_prior.pt"
    ckpt_base = torch.load(baseline_ckpt_path, map_location=device)

    grn = GRNGNN_V5(
        n_regulators=n_regulators,
        n_genes=ckpt_base["n_genes"], 
        n_layers=3,
        gene_emb_dim=64,
        hidden_dim=128,
        d_nexus=cfg.hidden_size, # Recursive Injection Dim
        use_dose=False
    ).to(device)
    
    # Check if we have V4 weights, otherwise warm-start from Baseline
    v4_ckpt_path = "/home/machina/MANTRA/experiments/nexus/v4_recursive_injection/checkpoints/model_v4_recursive.pt"
    if Path(v4_ckpt_path).exists():
        ckpt_v4 = torch.load(v4_ckpt_path, map_location=device)
        grn.load_state_dict(ckpt_v4["grn_state"], strict=False)
        print("Warm-started GNN from V4 Recursive Weights.")
    else:
        grn.load_state_dict(ckpt_base["model_state_dict"], strict=False)
        print("Warm-started GNN from Baseline Energy Weights.")
    
    class_map_path = "/home/machina/MANTRA/experiments/nexus/v3_functional_slotting/models/token_to_class_id.npy"
    adapter = NexusAdapter_V5(cfg, tokenizer, class_map_path=class_map_path).to(device)
    
    # Warm-start GeneCAM from V3 weights (High-Res)
    # Warm-start PathwayCAM from V3 weights (Baseline)
    # ... logic for separate manifold loading
    
    W = torch.from_numpy(ckpt_v4.get("W", ckpt_v4.get("grn_state", {}).get("W", np.zeros((100, 100))))).to(device) # We should load from the actual source
    # Better: Load from the original grn_k562_energy_prior.pt
    baseline_ckpt_path = "/home/machina/MANTRA/out/models/grn/hvg100/grn_k562_energy_prior.pt"
    ckpt_base = torch.load(baseline_ckpt_path, map_location=device)
    W = torch.from_numpy(ckpt_base["W"]).to(device)
    A = torch.from_numpy(ckpt_base["A"]).to(device)
    
    optimizer = optim.Adam([
        {"params": grn.parameters()},
        {"params": adapter.parameters()}
    ], lr=cfg.lr)
    
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    grn.train()
    adapter.train()
    adapter.model.cam.init_state(cfg.batch_size, device)
    
    print("Starting Phase 3 Multi-Scale Hierarchical Training...")
    train_log = open(exp_dir / "training_log.txt", "w")
    
    for epoch in range(cfg.max_epochs):
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            reg_idx = batch["reg_idx"].to(device)
            dose = batch["dose"].to(device)
            target_obs = batch["target"].to(device) # DeltaP [B, K]
            
            # Step A: Multi-Scale Nexus Retrieval
            nexus_outputs = adapter(reg_idx, dose)
            nexus_signal = nexus_outputs["nexus_signal"] # [B, D]
            deltaP_corr_global = nexus_outputs["deltaP_corr"] # [B, K]
            deltaY_pred = nexus_outputs["deltaY_pred"] # [B, 3]
            
            # Step B: Recursive GNN Forward (V5)
            deltaE_pred = grn(reg_idx, dose, A, nexus_signal=nexus_signal)
            deltaP_gnn = deltaE_pred @ W
            
            # Total Prediction
            deltaP_final = deltaP_gnn + deltaP_corr_global
            
            loss_p = criterion(deltaP_final, target_obs)
            
            # SMR Loss
            target_y = batch.get("target_y")
            if target_y is not None:
                target_y = target_y.to(device)
                loss_y = criterion(deltaY_pred, target_y)
                sign_acc = (torch.sign(deltaY_pred) == torch.sign(target_y)).float().mean()
            else:
                loss_y = 0.0
                sign_acc = torch.tensor(0.0)
                
            loss = loss_p + 1.0 * loss_y # Joint Optimization
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=f"{loss.item():.6f}", trait_acc=f"{sign_acc.item():.2%}")
            
            if i % 100 == 0:
                train_log.write(f"Step {i}, Loss: {loss.item():.6f}, TraitAcc: {sign_acc.item():.4f}\n")
                train_log.flush()

    train_log.close()
            
    # 4. Save
    out_dir = exp_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "grn_state": grn.state_dict(),
        "adapter_state": adapter.state_dict(),
        "cfg": cfg
    }, out_dir / "model_v5_multiscale.pt")
    print(f"V5 Multi-Scale Model Saved to {out_dir}")

if __name__ == "__main__":
    train_v5()
