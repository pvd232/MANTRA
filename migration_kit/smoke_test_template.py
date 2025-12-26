import torch
import torch.nn as nn
import sys
import os

# =============================================================================
# 💨 SMOKE TEST TEMPLATE
# =============================================================================
# USAGE:
# 1. Copy this file to tests/smoke_test_v[XX].py
# 2. Update the imports to point to your new model
# 3. Adjust the hyperparameters in run_smoke_test()
# 4. Run: python tests/smoke_test_v[XX].py
#
# GOAL:
# Ensure the model initializes, runs a forward pass, and backpropagates
# without crashing, NaN, or shape mismatches.
# =============================================================================

# Add experiment model directory to path (adjust "v68" as needed)
sys.path.append(os.path.join(os.path.dirname(__file__), "../experiments/v68_lane_unification/models"))

# --- REPLACE WITH YOUR MODEL IMPORT ---
# from hybrid_transformer import HybridTransformer
# --------------------------------------

def run_smoke_test():
    print("🚀 Starting Smoke Test...")
    
    # 1. Configuration (Minimal)
    config = {
        'vocab_size': 1000,
        'dim': 64,
        'depth': 2,
        'heads': 4,
        'context_len': 128,
        'use_memory': True
    }
    
    # 2. Initialization
    print("   [1/4] Initializing model...", end=" ")
    try:
        # model = HybridTransformer(**config)
        # model = model.cuda() if torch.cuda.is_available() else model
        print("✅ Passed")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return

    # 3. Dummy Data
    B, T = 2, 64
    x = torch.randint(0, config['vocab_size'], (B, T))
    if torch.cuda.is_available(): x = x.cuda()

    # 4. Forward Pass
    print("   [2/4] Forward pass...", end=" ")
    try:
        # logits, aux_loss = model(x)
        # assert logits.shape == (B, T, config['vocab_size']), f"Shape Mismatch: {logits.shape}"
        print("✅ Passed")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return

    # 5. Backward Pass
    print("   [3/4] Backward pass (gradients)...", end=" ")
    try:
        # loss = logits.mean() + aux_loss
        # loss.backward()
        print("✅ Passed")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return

    print("\n✨ Smoke Test Complete! You are ready to train.")

if __name__ == "__main__":
    run_smoke_test()
