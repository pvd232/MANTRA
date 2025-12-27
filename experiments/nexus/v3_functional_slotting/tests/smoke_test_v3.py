import torch
import sys
from pathlib import Path

# Add model to path
sys.path.append("/home/machina/MANTRA/experiments/nexus/v3_functional_slotting/models")
from nexus_v3 import NexusV3

def smoke_test():
    vocab_size = 15000
    hidden_size = 128
    class_map_path = "/home/machina/MANTRA/experiments/nexus/v3_functional_slotting/models/token_to_class_id.npy"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Initializing NexusV3...")
    model = NexusV3(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        class_map_path=class_map_path
    ).to(device)
    
    print("Model initialized successfully.")
    
    # Check if class_lookup is populated
    lookup = model.cam.class_lookup
    n_reg = 9866
    reg_lookup = lookup[:n_reg]
    print(f"Class lookup head: {reg_lookup[:10].tolist()}")
    
    # Verify bucketing
    x = torch.tensor([[0, 1, 2], [10, 11, 12]], device=device) # [B, L]
    # Token 0 and 10 might be in different classes
    
    # Simulate a forward pass or just a read
    h = torch.randn(2, 3, hidden_size, device=device)
    slot_values = model.cam.slot_values.expand(2, -1, -1)
    slot_keys = model.cam.slot_keys.expand(2, -1, -1)
    
    print("Testing CAM read...")
    val_out, max_sim, diag = model.cam.read(h, slot_values, slot_keys, tids=x)
    buckets = diag["buckets"]
    print(f"Tokens: {x[0].tolist()} -> Buckets: {buckets[0].tolist()}")
    
    # Check if tokens in same class land in same bucket
    # Let's find two tokens in the same class
    token_to_class = lookup.cpu().numpy()
    unique, counts = np.unique(token_to_class[:n_reg], return_counts=True)
    shared_class = unique[counts > 1][0]
    tokens_in_class = np.where(token_to_class[:n_reg] == shared_class)[0]
    
    if len(tokens_in_class) > 1:
        t1, t2 = tokens_in_class[0], tokens_in_class[1]
        # Match batch size 2 and seq len 2
        x_shared = torch.tensor([[t1, t2], [t1, t2]], device=device)
        # Use query with batch size 2 and seq len 2
        _, _, diag_shared = model.cam.read(h[:, :2], slot_values, slot_keys, tids=x_shared)
        b1, b2 = diag_shared["buckets"][0][0].item(), diag_shared["buckets"][0][1].item()
        print(f"Class {shared_class}: Token {t1} -> Bucket {b1}, Token {t2} -> Bucket {b2}")
        assert b1 == b2, f"Buckets should be identical for same class: {b1} != {b2}"
        print("Success: Functional bucketing confirmed.")
    else:
        print("No shared classes found to verify bucketing.")

if __name__ == "__main__":
    import numpy as np
    smoke_test()
