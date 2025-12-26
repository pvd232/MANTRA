from dataclasses import dataclass, field
from typing import Optional

@dataclass
class NexusConfig:
    # Model architecture
    vocab_size: int = 2000 # To be updated by tokenizer
    hidden_size: int = 256
    num_layers: int = 4
    n_buckets: int = 512
    slots_per_bucket: int = 32
    persistent_slots: int = 31
    
    # Nexus specifics
    ema_alpha: float = 0.01
    regret_gamma: float = 1.0
    alpha: float = 1.0 # Semantic/lexical blend
    tau: float = 0.1 # Attention temperature
    
    # Training / Mode B
    top_p: int = 16 # Number of programs in record
    max_seq_len: int = 32
    lr: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 10
    
    # Bio-specifics
    n_programs: int = 100 # To be updated
    n_regulators: int = 1000 # To be updated
