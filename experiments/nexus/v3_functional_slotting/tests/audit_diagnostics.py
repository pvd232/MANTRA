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
from nexus_v3 import NexusV3 # Ensure V3 is in path
from mantra.nexus.tokenizer import MLFGTokenizer
from mantra.nexus.datasets import MLFGTokenDataset
from mantra.nexus.configs import NexusConfig
from mantra.grn.models import GRNGNN
from mantra.grn.dataset import K562RegDeltaDataset

def run_diagnostics(exp_v="v3_1"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path("/home/machina/MANTRA/experiments/nexus/v3_functional_slotting")
    
    # 1. Config
    ckpt_name = "model_v3_1_hires.pt" if exp_v == "v3_1" else "model_v3_alpha08.pt"
    n_buckets = 4096 if exp_v == "v3_1" else 512
    slots_per_bucket = 4 if exp_v == "v3_1" else 32
    
    vocab_path = exp_dir / "checkpoints/vocab.json"
    with open(vocab_path, "r") as f:
        meta = json.load(f)
        
    cfg = NexusConfig(
        vocab_size=meta["vocab_size"],
        n_regulators=meta["n_regulators"],
        n_programs=meta["n_programs"],
        alpha=0.8,
        n_buckets=n_buckets,
        slots_per_bucket=slots_per_bucket
    )
    tokenizer = MLFGTokenizer(n_regulators=cfg.n_regulators, n_programs=cfg.n_programs)
    
    class_map_path = str(exp_dir / "models/token_to_class_id.npy")
    adapter = NexusAdapter(cfg, tokenizer, class_map_path=class_map_path).to(device)
    
    # Load weights
    ckpt_path = exp_dir / "checkpoints" / ckpt_name
    adapter.load_state_dict(torch.load(ckpt_path, map_location=device))
    adapter.eval()
    
    # 2. Setup Data
    val_npz = "/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/val.npz"
    val_ds = K562RegDeltaDataset(Path(val_npz))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    # Baseline for MSE
    baseline_ckpt_path = "/home/machina/MANTRA/out/models/grn/hvg100/grn_k562_energy_prior.pt"
    ckpt_base = torch.load(baseline_ckpt_path, map_location=device)
    grn = GRNGNN(
        n_regulators=ckpt_base["n_regulators"],
        n_genes=ckpt_base["n_genes"],
        **{k: v for k, v in ckpt_base["grn_model_cfg"].items() if k in ["n_layers", "gene_emb_dim", "hidden_dim", "dropout", "use_dose"]}
    ).to(device)
    grn.load_state_dict(ckpt_base["model_state_dict"])
    grn.eval()
    W = torch.from_numpy(ckpt_base["W"]).to(device)
    A = torch.from_numpy(ckpt_base["A"]).to(device)

    # 3. Warming Pass & Diagnostics
    print(f"Running Diagnostic Audit for {exp_v}...")
    train_npz = "/home/machina/MANTRA/data/interim/grn_k562_gwps_hvg100_npz/train.npz"
    train_ds = MLFGTokenDataset(Path(train_npz), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    
    gate_activation_sums = 0
    total_tokens = 0
    
    # Reset memory for diagnostic run
    adapter.model.cam.init_state(1, device)
    
    with torch.no_grad():
        # Limit to 1000 samples for faster representative audit
        subset_loader = list(train_loader)[:1000]
        for batch in tqdm.tqdm(subset_loader, desc="Diagnostic Pass (Train Subset)"):
            input_ids = batch["input_ids"].to(device)
            # Forward returns: logits, h_fused_all, _, _, _, final_masks
            out = adapter.model(input_ids)
            masks = out[-1] # [B, L, 1]
            gate_activation_sums += masks.sum().item()
            total_tokens += masks.numel()
            
    gate_density = gate_activation_sums / total_tokens
    
    # Post-Pass Memory Audit
    slot_tids = adapter.model.cam.slot_tids.squeeze(0) # [total_slots]
    populated_mask = (slot_tids != -1)
    slot_utilization = populated_mask.float().mean().item()
    
    # Class Coverage
    class_lookup = adapter.model.cam.class_lookup
    # Get distinct functional classes in data
    # We can check which class_ids are present in populated slots
    # class_lookup maps token_ids -> class_ids
    populated_tids = slot_tids[populated_mask].cpu().numpy()
    unique_populated_tids = np.unique(populated_tids)
    
    # We need to know the total unique classes available for regulators
    class_map = class_lookup.cpu().numpy()
    total_classes = len(np.unique(class_map[:meta["n_regulators"]]))
    
    populated_classes = len(np.unique(class_map[unique_populated_tids]))
    class_coverage = populated_classes / total_classes

    # 4. Evaluation (MSE)
    mse_total = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc="Eval (Val)"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            deltaE_pred = grn(batch["reg_idx"], batch.get("dose"), A=A)
            deltaP_baseline = deltaE_pred @ W
            deltaP_corr = adapter(batch["reg_idx"], batch.get("dose"))
            deltaP_final = deltaP_baseline + deltaP_corr
            mse_total += ((deltaP_final - batch["deltaP_obs"])**2).mean().item()
    mse_avg = mse_total / len(val_loader)

    print(f"\n--- Diagnostic Results ({exp_v}) ---")
    print(f"Gate Density (Write Freq): {gate_density:.4%}")
    print(f"Slot Utilization:         {slot_utilization:.4%}")
    print(f"Class Coverage:           {class_coverage:.4%} ({populated_classes}/{total_classes})")
    print(f"Final MSE:                {mse_avg:.6f}")
    
    return {
        "gate_density": gate_density,
        "slot_utilization": slot_utilization,
        "class_coverage": class_coverage,
        "mse": mse_avg
    }

if __name__ == "__main__":
    results_v3 = run_diagnostics("v3")
    results_v3_1 = run_diagnostics("v3_1")
    
    with open("/home/machina/MANTRA/experiments/nexus/v3_functional_slotting/audits/diagnostic_report.json", "w") as f:
        json.dump({"v3": results_v3, "v3_1": results_v3_1}, f, indent=2)
