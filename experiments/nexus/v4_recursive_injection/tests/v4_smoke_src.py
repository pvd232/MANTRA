# experiments/nexus/v4_recursive_injection/tests/v4_smoke_src.py
"""
Smoke test to verify that the PROMOTED code in src/mantra works correctly.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Ensure src is in path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from mantra.grn.models import GRNGNN
from mantra.nexus.adapter import NexusAdapter
from mantra.nexus.configs import NexusConfig
from mantra.nexus.tokenizer import MLFGTokenizer

def smoke_test_src():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running smoke test on {device}...")

    # 1. Initialize Components from src
    n_regulators = 9866
    n_genes = 100
    n_programs = 30
    
    tokenizer = MLFGTokenizer(n_regulators=n_regulators, n_programs=n_programs)
    
    cfg = NexusConfig(
        vocab_size=tokenizer.vocab_size,
        n_programs=n_programs,
        hidden_size=256,
        num_layers=4
    )
    
    adapter = NexusAdapter(cfg, tokenizer).to(device)
    grn = GRNGNN(
        n_regulators=n_regulators,
        n_genes=n_genes,
        d_nexus=cfg.hidden_size
    ).to(device)
    
    # 2. Mock Input
    reg_idx = torch.tensor([0, 1, 2]).to(device)
    dose = torch.tensor([0.1, 0.5, 0.9]).to(device)
    A = torch.eye(n_genes).to(device)
    
    # 3. Forward Pass
    print("Testing NexusAdapter...")
    nexus_out = adapter(reg_idx=reg_idx, dose=dose)
    assert "nexus_signal" in nexus_out
    assert "deltaP_corr" in nexus_out
    print("NexusAdapter: OK")
    
    print("Testing GRNGNN with Recursive Injection...")
    deltaE_pred = grn(
        reg_idx=reg_idx, 
        dose=dose, 
        A=A, 
        nexus_signal=nexus_out["nexus_signal"]
    )
    assert deltaE_pred.shape == (3, n_genes)
    print("GRNGNN: OK")
    
    print("\n--- Smoke Test PASSED ---")

if __name__ == "__main__":
    smoke_test_src()
