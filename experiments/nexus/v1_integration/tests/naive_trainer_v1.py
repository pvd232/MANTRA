# Naive V1 Trainer (Snapshot for documentation of pathologies)
# This version contained the "Index Cheating" and "Double Counting" flaws.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mantra.nexus.adapter import NexusAdapter
from mantra.nexus.tokenizer import MLFGTokenizer

def train_naive():
    # ... setup code ...
    # [FLAW] Predict from the very last token
    out = adapter.model(input_ids)
    h_fused = out[1]
    h_last = h_fused[:, -1, :] # <--- FLAW: Lookahead to target tokens
    pred = adapter.proj(h_last)
    
    # [FLAW] No residual calculation (predicting total signal)
    loss = criterion(pred, target_obs) 
    loss.backward()
    # ...
