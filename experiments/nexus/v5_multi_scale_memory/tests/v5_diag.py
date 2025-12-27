import torch
import numpy as np
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
from adapter_v5 import NexusAdapter_V5
from mantra.nexus.tokenizer import MLFGTokenizer
from mantra.nexus.datasets import MLFGTokenDataset
from mantra.nexus.configs import NexusConfig

def run_diag_v5():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path(__file__).parent.parent
    
    ckpt_path = exp_dir / "checkpoints/model_v5_multiscale.pt"
    if not ckpt_path.exists():
        print(f"V5 Checkpoint not found at {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    tokenizer = MLFGTokenizer(n_regulators=cfg.n_regulators, n_programs=cfg.n_programs)
    
    adapter = NexusAdapter_V5(cfg, tokenizer, class_map_path="/home/machina/MANTRA/experiments/nexus/v3_functional_slotting/models/token_to_class_id.npy").to(device)
    adapter.load_state_dict(ckpt["adapter_state"])
    adapter.eval()
    
    val_loader = DataLoader(MLFGTokenDataset("/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/val.npz", tokenizer), batch_size=1, shuffle=False)
    
    gates = []
    w_fine = []
    w_coarse = []
    sims_fine = []
    sims_coarse = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(val_loader, desc="Diagnosing")):
            input_ids = batch["input_ids"].to(device)
            # Inspect internal state
            logits, h_fused, _, _, _, _ = adapter.model(input_ids)
            # Re-read to get diag
            _, _, diag = adapter.model.cam.read(h_fused, input_ids)
            
            # fusion gate is [B, T, D]
            g = diag["gate"][:, 2, :].mean().item()
            gates.append(g)
            
            # sim values
            sim_f = diag["sim_fine"][:, 2].item()
            sim_c = diag["sim_coarse"][:, 2].item()
            sims_fine.append(sim_f)
            sims_coarse.append(sim_c)

            # Measure write gates by simulating update_step on val (using same logic)
            # (In practice, we want to see how often it *would* write)
            write_diag = adapter.model.cam.update_step(h_fused, input_ids, h_fused, logits, diag)
            w_fine.append(write_diag["gate_fine"])
            w_coarse.append(write_diag["gate_coarse"])
            
            if i > 500: break
            
    print(f"\n--- V5 Hierarchical Diagnostic ---")
    print(f"Mean Fusion Gate (0=Coarse, 1=Fine): {np.mean(gates):.4f}")
    print(f"Mean Write Density (Fine):           {np.mean(w_fine):.4%}")
    print(f"Mean Write Density (Coarse):         {np.mean(w_coarse):.4%}")
    print(f"Mean Similarity (Fine):              {np.mean(sims_fine):.4f}")
    print(f"Mean Similarity (Coarse):            {np.mean(sims_coarse):.4f}")
    
    # Histogram-like summary
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(gates, bins=bins)
    print("\nGate Distribution (0.0 to 1.0):")
    print(hist)

if __name__ == "__main__":
    run_diag_v5()
