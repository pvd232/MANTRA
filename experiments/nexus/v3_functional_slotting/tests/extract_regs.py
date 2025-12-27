import torch
import numpy as np
from pathlib import Path

def extract_regulators():
    ckpt_path = "/home/machina/MANTRA/out/models/eggfm/eggfm_energy_k562_hvg_hvg75.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Actually, the regulators are the features in the raw GWPS h5ad.
    # In the NPZ, reg_idx refers to indices in regs_pert.
    # Let's see if we can get the regulator list from another source or by inspecting the h5ad.
    import scanpy as sc
    ad = sc.read_h5ad("data/raw/k562_gwps.h5ad", backed="r")
    regs = ad.obs["gene"].unique().tolist()
    print(f"Total unique regulators: {len(regs)}")
    print(f"Examples: {regs[:50]}")

if __name__ == "__main__":
    extract_regulators()
