#!/usr/bin/env python3
# experiments/nexus/v4_recursive_injection/tests/v4_smr_eval.py
import torch
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Local imports
sys.path.append("/home/machina/MANTRA/src")
sys.path.append(str(Path(__file__).parent.parent / "models"))
from adapter_v4 import NexusAdapter_V4
from v4_models import GRNGNN_V4
from mantra.nexus.tokenizer import MLFGTokenizer
from mantra.nexus.datasets import MLFGTokenDataset
from mantra.grn.dataset import K562RegDeltaDataset

def run_smr_eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Baseline
    baseline_ckpt_path = "/home/machina/MANTRA/out/models/grn/hvg100/grn_k562_energy_prior.pt"
    ckpt_base = torch.load(baseline_ckpt_path, map_location=device)
    W = torch.from_numpy(ckpt_base["W"]).to(device)
    A = torch.from_numpy(ckpt_base["A"]).to(device)
    
    # 2. Load Nexus V4
    v4_ckpt_path = "/home/machina/MANTRA/experiments/nexus/v4_recursive_injection/checkpoints/model_v4_recursive.pt"
    ckpt_v4 = torch.load(v4_ckpt_path, map_location=device)
    
    n_regulators = 9866
    tokenizer = MLFGTokenizer(n_regulators=n_regulators, n_programs=ckpt_v4["cfg"].n_programs)
    adapter = NexusAdapter_V4(ckpt_v4["cfg"], tokenizer).to(device)
    adapter.load_state_dict(ckpt_v4["adapter_state"])
    
    grn = GRNGNN_V4(
        n_regulators=n_regulators,
        n_genes=ckpt_base["n_genes"],
        d_nexus=ckpt_v4["cfg"].hidden_size
    ).to(device)
    grn.load_state_dict(ckpt_v4["grn_state"])
    
    adapter.eval(); grn.eval()
    
    # 3. Load SMR Readouts (Theta)
    theta_mch = torch.from_numpy(np.load("/home/machina/MANTRA/out/smr/theta_MCH.npy")).to(device).float()
    
    # 4. Load Data
    val_npz = "/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/val.npz"
    dataset = K562RegDeltaDataset(Path(val_npz))
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    total_sign_correct_base = 0.0
    total_sign_correct_v4 = 0.0
    n = 0
    
    with torch.no_grad():
        for batch in loader:
            reg_idx = batch["reg_idx"].to(device)
            target_p = batch["deltaP_obs"].to(device)
            # Oracle Trait Delta = Observed Programs @ Theta
            target_y = target_p @ theta_mch
            
            # Baseline Prediction
            from mantra.grn.models import GRNGNN
            grn_cfg = {k: v for k, v in ckpt_base["grn_model_cfg"].items() if k in ["n_layers", "gene_emb_dim", "hidden_dim", "dropout", "use_dose"]}
            base_model = GRNGNN(n_regulators=ckpt_base["n_regulators"], n_genes=ckpt_base["n_genes"], **grn_cfg).to(device)
            base_model.load_state_dict(ckpt_base["model_state_dict"])
            base_model.eval()
            
            deltaE_base = base_model(reg_idx, None, A)
            deltaP_base = deltaE_base @ W
            deltaY_base = deltaP_base @ theta_mch
            
            # V4 Prediction
            nexus_out = adapter(reg_idx, None)
            deltaE_v4 = grn(reg_idx, None, A, nexus_signal=nexus_out["nexus_signal"])
            deltaP_v4 = deltaE_v4 @ W + nexus_out["deltaP_corr"]
            deltaY_v4 = deltaP_v4 @ theta_mch
            
            # Sign Accuracy
            total_sign_correct_base += (torch.sign(deltaY_base) == torch.sign(target_y)).float().mean().item()
            total_sign_correct_v4 += (torch.sign(deltaY_v4) == torch.sign(target_y)).float().mean().item()
            n += 1
            
    avg_acc_base = total_sign_correct_base / n
    avg_acc_v4 = total_sign_correct_v4 / n
    
    print(f"--- V4 SMR (Trait) Fidelity Audit ---")
    print(f"Trait: MCH")
    print(f"Baseline Sign-Acc: {avg_acc_base:.4%}")
    print(f"Nexus V4 Sign-Acc: {avg_acc_v4:.4%}")
    print(f"Trait Fidelity Gain: {((avg_acc_v4 - avg_acc_base) / avg_acc_base) * 100:.2f}%")

if __name__ == "__main__":
    run_smr_eval()
