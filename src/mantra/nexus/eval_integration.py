import torch
import numpy as np
import json
import tqdm
from pathlib import Path
from mantra.grn.models import GRNGNN, TraitHead
from mantra.grn.config import GRNTrainConfig, GRNLossConfig
from mantra.grn.trainer import compute_grn_losses
from mantra.nexus.adapter import NexusAdapter
from mantra.nexus.tokenizer import MLFGTokenizer
from mantra.nexus.datasets import MLFGTokenDataset
from mantra.nexus.configs import NexusConfig
from mantra.grn.dataset import K562RegDeltaDataset
from torch.utils.data import DataLoader

def run_eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data
    val_npz = "/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/val.npz"
    dataset = K562RegDeltaDataset(Path(val_npz))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # 2. Load Models
    baseline_ckpt_path = "/home/machina/MANTRA/out/models/grn/hvg100/grn_k562_energy_prior.pt"
    ckpt_base = torch.load(baseline_ckpt_path, map_location=device)
    
    grn_cfg = {k: v for k, v in ckpt_base["grn_model_cfg"].items() if k in ["n_layers", "gene_emb_dim", "hidden_dim", "dropout", "use_dose"]}
    
    grn = GRNGNN(
        n_regulators=ckpt_base["n_regulators"],
        n_genes=ckpt_base["n_genes"],
        **grn_cfg
    ).to(device)
    grn.load_state_dict(ckpt_base["model_state_dict"])
    grn.eval()
    
    W = torch.from_numpy(ckpt_base["W"]).to(device)
    A = torch.from_numpy(ckpt_base["A"]).to(device)
    
    # Load Trained Nexus
    with open("/home/machina/MANTRA/out/nexus/vocab.json", "r") as f:
        meta = json.load(f)
        
    cfg = NexusConfig(
        vocab_size=meta["vocab_size"],
        n_regulators=meta["n_regulators"],
        n_programs=meta["n_programs"]
    )
    tokenizer = MLFGTokenizer(n_regulators=cfg.n_regulators, n_programs=cfg.n_programs)
    nexus = NexusAdapter(cfg, tokenizer).to(device)
    
    # 1. Initialize persistent memory buffers for eval batch size
    nexus.model.cam.init_state(1, device) 
    nexus.load_state_dict(torch.load("/home/machina/MANTRA/out/nexus/model.pt", map_location=device))
    nexus.eval()
    
    # 2. Memory Warming Pass (Warming the manifold with training signatures)
    print("Warming Nexus memory with training data (sequential)...", flush=True)
    train_npz = "/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/train.npz"
    train_ds = MLFGTokenDataset(Path(train_npz), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(train_loader, desc="Warming"):
            input_ids = batch["input_ids"].to(device)
            _ = nexus.model(input_ids) 
    
    print("Warming complete. Running evaluation...", flush=True)
    
    # 3. Evaluation Loop
    results = []
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Baseline Prediction (GRN only)
            deltaE_pred = grn(batch["reg_idx"], batch.get("dose"), A=A)
            deltaP_baseline = deltaE_pred @ W
            
            # Nexus Correction
            deltaP_corr = nexus(batch["reg_idx"], batch.get("dose"))
            deltaP_final = deltaP_baseline + deltaP_corr
            
            # Target
            deltaP_obs = batch["deltaP_obs"]
            
            # Metrics
            mse_baseline = ((deltaP_baseline - deltaP_obs)**2).mean().item()
            mse_nexus = ((deltaP_final - deltaP_obs)**2).mean().item()
            
            mag_base = torch.norm(deltaP_baseline, dim=1).mean().item()
            mag_obs = torch.norm(deltaP_obs, dim=1).mean().item()
            mag_corr = torch.norm(deltaP_corr, dim=1).mean().item()
            
            results.append({
                "mse_baseline": mse_baseline,
                "mse_nexus": mse_nexus,
                "mag_base": mag_base,
                "mag_obs": mag_obs,
                "mag_corr": mag_corr,
                "reg_idx": batch["reg_idx"].cpu().numpy()
            })
            
    # 4. Reporting
    avg_mse_b = np.mean([r["mse_baseline"] for r in results])
    avg_mse_n = np.mean([r["mse_nexus"] for r in results])
    avg_mag_b = np.mean([r["mag_base"] for r in results])
    avg_mag_o = np.mean([r["mag_obs"] for r in results])
    avg_mag_c = np.mean([r["mag_corr"] for r in results])
    
    print(f"--- Performance Summary ---")
    print(f"Baseline MSE: {avg_mse_b:.4f}")
    print(f"Nexus+ MSE:    {avg_mse_n:.4f}")
    print(f"Improvement:   {(avg_mse_b - avg_mse_n)/avg_mse_b * 100:.2f}%")
    print(f"--- Magnitude Stats ---")
    print(f"Avg Norm (Baseline): {avg_mag_b:.4f}")
    print(f"Avg Norm (Observed): {avg_mag_o:.4f}")
    print(f"Avg Norm (Nexus Corr): {avg_mag_c:.4f}")

if __name__ == "__main__":
    # Mocking GRNGNN if not imported correctly or need specific architecture
    # Assume it's available in mantra.grn.models
    run_eval()
