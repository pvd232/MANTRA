import torch
import numpy as np
import json
import tqdm
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Local imports
sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent.parent / "v3_functional_slotting/models"))

from v4_models import GRNGNN_V4
from adapter_v4 import NexusAdapter_V4
from mantra.nexus.tokenizer import MLFGTokenizer
from mantra.nexus.datasets import MLFGTokenDataset
from mantra.nexus.configs import NexusConfig

def run_eval_v4():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path(__file__).parent.parent
    
    # 1. Load Data
    val_npz = "/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/val.npz"
    from mantra.grn.dataset import K562RegDeltaDataset
    dataset = K562RegDeltaDataset(Path(val_npz))
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 2. Load V4 Model
    ckpt_path = exp_dir / "checkpoints/model_v4_recursive.pt"
    if not ckpt_path.exists():
        print(f"V4 Checkpoint not found at {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    tokenizer = MLFGTokenizer(n_regulators=cfg.n_regulators, n_programs=cfg.n_programs)
    
    # Load Baseline Matrices
    baseline_ckpt_path = "/home/machina/MANTRA/out/models/grn/hvg100/grn_k562_energy_prior.pt"
    ckpt_base = torch.load(baseline_ckpt_path, map_location=device)
    W = torch.from_numpy(ckpt_base["W"]).to(device)
    A = torch.from_numpy(ckpt_base["A"]).to(device)
    
    adapter = NexusAdapter_V4(cfg, tokenizer, class_map_path="/home/machina/MANTRA/experiments/nexus/v3_functional_slotting/models/token_to_class_id.npy").to(device)
    adapter.load_state_dict(ckpt["adapter_state"])
    adapter.eval()
    
    grn = GRNGNN_V4(
        n_regulators=cfg.n_regulators,
        n_genes=ckpt_base["n_genes"], # Align with baseline
        n_layers=3,
        gene_emb_dim=64,
        hidden_dim=128,
        d_nexus=cfg.hidden_size,
        use_dose=False
    ).to(device)
    grn.load_state_dict(ckpt["grn_state"])
    grn.eval()
    
    # 3. Warming Pass
    print("Warming Nexus memory for V4 Audit...")
    train_npz = "/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/train.npz"
    train_ds = MLFGTokenDataset(Path(train_npz), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    
    adapter.model.cam.init_state(1, device)
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(train_loader, desc="Warming", total=1000)): # Subset for audit
            _ = adapter.model(batch["input_ids"].to(device), start_pos=0) # Call model directly
            if i > 1000: break # Just warm up first 1000
            
    # 4. Evaluation Loop
    results = []
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Nexus Retrieval
            nexus_outputs = adapter(batch["reg_idx"], batch.get("dose"))
            nexus_signal = nexus_outputs["nexus_signal"]
            deltaP_corr = nexus_outputs["deltaP_corr"]
            
            # Baseline Prediction (using V4 GNN without signal for reference?)
            # No, using original baseline for absolute gain comparison.
            deltaE_base = grn(batch["reg_idx"], batch.get("dose"), A, nexus_signal=None)
            deltaP_baseline = deltaE_base @ W
            
            # Recursive Prediction
            deltaE_rec = grn(batch["reg_idx"], batch.get("dose"), A, nexus_signal=nexus_signal)
            deltaP_gnn = deltaE_rec @ W
            deltaP_final = deltaP_gnn + deltaP_corr
            
            # Target
            deltaP_obs = batch["deltaP_obs"]
            
            mse_baseline = ((deltaP_baseline - deltaP_obs)**2).mean().item()
            mse_v4 = ((deltaP_final - deltaP_obs)**2).mean().item()
            
            results.append({"mse_b": mse_baseline, "mse_v4": mse_v4})
            
    avg_b = np.mean([r["mse_b"] for r in results])
    avg_v4 = np.mean([r["mse_v4"] for r in results])
    
    print(f"\n--- V4 Performance Audit (Recursive Joint) ---")
    print(f"Baseline MSE:  {avg_b:.6f}")
    print(f"Nexus V4 MSE:  {avg_v4:.6f}")
    print(f"Recursive Gain: {(avg_b - avg_v4)/avg_b * 100:.4f}%")

if __name__ == "__main__":
    run_eval_v4()
