import torch
import numpy as np
import json
import tqdm
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Local imports
sys.path.append("/home/machina/MANTRA/src")
sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent.parent / "v3_functional_slotting/models"))
sys.path.append(str(Path(__file__).parent.parent.parent / "v4_recursive_injection/models"))

from v5_models import NexusV5
from v5_gnn import GRNGNN_V5
from adapter_v5 import NexusAdapter_V5
from mantra.nexus.tokenizer import MLFGTokenizer
from mantra.nexus.datasets import MLFGTokenDataset
from mantra.nexus.configs import NexusConfig

def run_eval_v5():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path(__file__).parent.parent
    
    # 1. Load Data
    val_npz = "/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/val.npz"
    from mantra.grn.dataset import K562RegDeltaDataset
    dataset = K562RegDeltaDataset(Path(val_npz))
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 2. Load V5 Model
    ckpt_path = exp_dir / "checkpoints/model_v5_multiscale.pt"
    if not ckpt_path.exists():
        print(f"V5 Checkpoint not found at {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    
    tokenizer = MLFGTokenizer(n_regulators=cfg.n_regulators, n_programs=cfg.n_programs)
    
    class_map_path = "/home/machina/MANTRA/experiments/nexus/v3_functional_slotting/models/token_to_class_id.npy"
    adapter = NexusAdapter_V5(cfg, tokenizer, class_map_path=class_map_path).to(device)
    adapter.load_state_dict(ckpt["adapter_state"])
    adapter.eval()
    
    # Load Baseline Matrices for n_genes
    baseline_ckpt_path = "/home/machina/MANTRA/out/models/grn/hvg100/grn_k562_energy_prior.pt"
    ckpt_base = torch.load(baseline_ckpt_path, map_location=device)
    W = torch.from_numpy(ckpt_base["W"]).to(device)
    A = torch.from_numpy(ckpt_base["A"]).to(device)
    
    grn = GRNGNN_V5(
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
    
    # 3. Warming Pass
    print("Warming Multi-Scale Memory for V5 Audit...")
    train_npz = "/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/train.npz"
    train_ds = MLFGTokenDataset(Path(train_npz), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    
    adapter.model.cam.init_state(1, device)
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(train_loader, desc="Warming", total=1000)):
            _ = adapter.model(batch["input_ids"].to(device), start_pos=0)
            if i > 1000: break
            
    # 4. Evaluation Loop
    results = []
    gate_activations = []
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Nexus Retrieval
            nexus_outputs = adapter(batch["reg_idx"], batch.get("dose"))
            nexus_signal = nexus_outputs["nexus_signal"]
            deltaP_corr = nexus_outputs["deltaP_corr"]
            
            # (Optional: track gate if available in outputs)
            # In adapter_v5 forward, let's extract masks/gate
            # For now we'll just check MSE
            
            # Baseline Prediction
            # Note: We should compare against the same baseline as V4
            deltaE_base = grn(batch["reg_idx"], batch.get("dose"), A, nexus_signal=None)
            deltaP_baseline = deltaE_base @ W
            
            # V5 Prediction
            deltaE_v5 = grn(batch["reg_idx"], batch.get("dose"), A, nexus_signal=nexus_signal)
            deltaP_gnn = deltaE_v5 @ W
            deltaP_final = deltaP_gnn + deltaP_corr
            
            # Target
            deltaP_obs = batch["deltaP_obs"]
            
            mse_baseline = ((deltaP_baseline - deltaP_obs)**2).mean().item()
            mse_v5 = ((deltaP_final - deltaP_obs)**2).mean().item()
            
            results.append({"mse_b": mse_baseline, "mse_v5": mse_v5})
            
    avg_b = np.mean([r["mse_b"] for r in results])
    avg_v5 = np.mean([r["mse_v5"] for r in results])
    
    print(f"\n--- V5 Performance Audit (Multi-Scale Hierarchical) ---")
    print(f"Baseline MSE:  {avg_b:.6f}")
    print(f"Nexus V5 MSE:  {avg_v5:.6f}")
    print(f"Total Gain:    {(avg_b - avg_v5)/avg_b * 100:.4f}%")

if __name__ == "__main__":
    run_eval_v5()
