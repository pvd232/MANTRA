import json
import numpy as np
import torch
from pathlib import Path

def finalize_class_lookup():
    # 1. Get raw regulators from data
    # (Matches make_grn_npz logic: sorted unique perturbed regs)
    import scanpy as sc
    ad = sc.read_h5ad("data/raw/k562_gwps.h5ad", backed="r")
    all_regs = ad.obs["gene"].to_numpy()
    control_value = "non-targeting"
    regs_pert = np.unique(all_regs[all_regs != control_value])
    
    # 2. Load the class map we built
    with open("/home/machina/MANTRA/experiments/nexus/v3_functional_slotting/models/regulator_to_class.json", 'r') as f:
        reg_to_class = json.load(f)
    
    # 3. Map each class name to a unique integer ID
    unique_classes = sorted(list(set(reg_to_class.values())))
    class_to_id = {cls: i for i, cls in enumerate(unique_classes)}
    
    # 4. Create the final lookup table for regulator tokens (0...N_REG-1)
    # Tokenizer map: token_id = index in regs_pert
    n_reg = len(regs_pert)
    token_to_class_id = np.zeros(n_reg, dtype=np.int64)
    for i, reg_name in enumerate(regs_pert):
        cls_name = reg_to_class.get(reg_name, "UNKNOWN")
        token_to_class_id[i] = class_to_id[cls_name]
    
    output_path = "/home/machina/MANTRA/experiments/nexus/v3_functional_slotting/models/token_to_class_id.npy"
    np.save(output_path, token_to_class_id)
    print(f"Saved token-to-class lookup for {n_reg} regulators to {output_path}")

if __name__ == "__main__":
    finalize_class_lookup()
