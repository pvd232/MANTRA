import torch
import numpy as np
import json
import tqdm
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Local imports
sys.path.append(str(Path(__file__).parent.parent / "models"))
from adapter_v3 import NexusAdapter
from mantra.nexus.tokenizer import MLFGTokenizer
from mantra.nexus.datasets import MLFGTokenDataset
from mantra.nexus.configs import NexusConfig
from mantra.grn.models import GRNGNN
from mantra.grn.dataset import K562RegDeltaDataset

def run_eval_v3_1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path(__file__).parent.parent
    
    # 1. Load Data
    val_npz = "/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/val.npz"
    dataset = K562RegDeltaDataset(Path(val_npz))
    
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
    
    # Load Trained Nexus V3.1
    vocab_path = exp_dir / "checkpoints/vocab.json"
    with open(vocab_path, "r") as f:
        meta = json.load(f)
        
    cfg = NexusConfig(
        vocab_size=meta["vocab_size"],
        n_regulators=meta["n_regulators"],
        n_programs=meta["n_programs"],
        alpha=0.8,
        n_buckets=4096, # High resolution
        slots_per_bucket=4
    )
    tokenizer = MLFGTokenizer(n_regulators=cfg.n_regulators, n_programs=cfg.n_programs)
    
    class_map_path = str(exp_dir / "models/token_to_class_id.npy")
    nexus = NexusAdapter(cfg, tokenizer, class_map_path=class_map_path).to(device)
    
    # Init and Load Weights
    nexus.model.cam.init_state(1, device) 
    nexus.load_state_dict(torch.load(exp_dir / "checkpoints/model_v3_1_hires.pt", map_location=device))
    nexus.eval()
    
    # 3. Memory Warming Pass
    print("Warming Nexus memory with training data (Hi-Res Manifold)...", flush=True)
    train_npz = "/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/train.npz"
    train_ds = MLFGTokenDataset(Path(train_npz), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(train_loader, desc="Warming"):
            input_ids = batch["input_ids"].to(device)
            _ = nexus.model(input_ids) 
    
    print("Warming complete. Running evaluation...", flush=True)
    
    # 4. Evaluation Loop
    results = []
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Baseline Prediction
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
            
            results.append({
                "mse_baseline": mse_baseline,
                "mse_nexus": mse_nexus,
                "reg_idx": batch["reg_idx"].cpu().numpy()
            })
            
    # 5. Reporting
    avg_mse_b = np.mean([r["mse_baseline"] for r in results])
    avg_mse_n = np.mean([r["mse_nexus"] for r in results])
    
    print(f"\n--- V3.1 Performance Summary (4096x4 Manifold) ---")
    print(f"Alpha:         {cfg.alpha}")
    print(f"Buckets:       {cfg.n_buckets}")
    print(f"Baseline MSE:  {avg_mse_b:.6f}")
    print(f"Nexus V3.1 MSE: {avg_mse_n:.6f}")
    print(f"Gain:          {(avg_mse_b - avg_mse_n)/avg_mse_b * 100:.4f}%")
    
    # Save Audits
    audit_data = {
        "avg_mse_baseline": float(avg_mse_b),
        "avg_mse_nexus": float(avg_mse_n),
        "gain_pct": float((avg_mse_b - avg_mse_n)/avg_mse_b * 100),
        "n_samples": len(results)
    }
    with open(exp_dir / "audits/eval_v3_1_results.json", "w") as f:
        json.dump(audit_data, f, indent=2)

if __name__ == "__main__":
    run_eval_v3_1()
