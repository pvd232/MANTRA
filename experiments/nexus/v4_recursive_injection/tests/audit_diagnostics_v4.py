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
from mantra.grn.dataset import K562RegDeltaDataset

def run_diagnostics_v4():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path(__file__).parent.parent
    
    # 1. Load V4 Model
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
        n_genes=ckpt_base["n_genes"],
        n_layers=3,
        gene_emb_dim=64,
        hidden_dim=128,
        d_nexus=cfg.hidden_size,
        use_dose=False
    ).to(device)
    grn.load_state_dict(ckpt["grn_state"])
    grn.eval()
    
    # 2. Setup Data
    val_npz = "/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/val.npz"
    val_ds = K562RegDeltaDataset(Path(val_npz))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    # 3. Warming Pass & Diagnostics
    print("Running Diagnostic Audit for V4...")
    train_npz = "/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/train.npz"
    train_ds = MLFGTokenDataset(Path(train_npz), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    
    gate_activation_sums = 0
    total_tokens = 0
    
    adapter.model.cam.init_state(1, device)
    with torch.no_grad():
        # Limit to 1000 samples
        for i, batch in enumerate(tqdm.tqdm(train_loader, desc="Diagnostic Pass (Train Subset)", total=1000)):
            input_ids = batch["input_ids"].to(device)
            logits, h_fused_all, _, _, _, final_masks = adapter.model(input_ids)
            gate_activation_sums += final_masks.sum().item()
            total_tokens += final_masks.numel()
            if i >= 999: break
            
    gate_density = gate_activation_sums / total_tokens
    
    slot_tids = adapter.model.cam.slot_tids.squeeze(0)
    populated_mask = (slot_tids != -1)
    slot_utilization = populated_mask.float().mean().item()
    
    class_lookup = adapter.model.cam.class_lookup
    populated_tids = slot_tids[populated_mask].cpu().numpy()
    unique_populated_tids = np.unique(populated_tids)
    
    class_map = class_lookup.cpu().numpy()
    total_classes = len(np.unique(class_map[:cfg.n_regulators]))
    populated_classes = len(np.unique(class_map[unique_populated_tids]))
    class_coverage = populated_classes / total_classes

    # 4. Evaluation (MSE)
    mse_total = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc="Eval (Val)"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            nexus_outputs = adapter(batch["reg_idx"], batch.get("dose"))
            nexus_signal = nexus_outputs["nexus_signal"]
            deltaP_corr = nexus_outputs["deltaP_corr"]
            
            deltaE_base = grn(batch["reg_idx"], batch.get("dose"), A, nexus_signal=None)
            deltaP_baseline = deltaE_base @ W
            
            deltaE_rec = grn(batch["reg_idx"], batch.get("dose"), A, nexus_signal=nexus_signal)
            deltaP_gnn = deltaE_rec @ W
            deltaP_final = deltaP_gnn + deltaP_corr
            
            mse_total += ((deltaP_final - batch["deltaP_obs"])**2).mean().item()
            
    mse_avg = mse_total / len(val_loader)

    print(f"\n--- Diagnostic Results (V4 Recursive) ---")
    print(f"Gate Density (Write Freq): {gate_density:.4%}")
    print(f"Slot Utilization:         {slot_utilization:.4%}")
    print(f"Class Coverage:           {class_coverage:.4%} ({populated_classes}/{total_classes})")
    print(f"Final MSE:                {mse_avg:.6f}")
    
    # Check baseline in this run for sanity
    # (Actually we know it from v4_eval.py)

if __name__ == "__main__":
    run_diagnostics_v4()
