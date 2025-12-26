import json
import torch
from typing import List, Dict, Optional

class MLFGTokenizer:
    """
    Tokenizer for MLFG perturbation records.
    Record format: <REG> <DOSE> <STATE> <PROG_K1, SIGN, BIN> ... <PROG_KP, SIGN, BIN> <ENDREC>
    """
    def __init__(
        self, 
        n_regulators: int, 
        n_programs: int, 
        n_dose_bins: int = 4, 
        n_state_bins: int = 1,
        n_mag_bins: int = 9
    ):
        self.n_regulators = n_regulators
        self.n_programs = n_programs
        self.n_dose_bins = n_dose_bins
        self.n_state_bins = n_state_bins
        self.n_mag_bins = n_mag_bins
        
        # Define offsets for token ranges
        self.reg_offset = 0
        self.dose_offset = self.reg_offset + n_regulators
        self.state_offset = self.dose_offset + n_dose_bins
        self.prog_offset = self.state_offset + n_state_bins
        
        # Each program has (2 signs * n_mag_bins) tokens
        # We index them as: prog_idx * (2 * n_mag_bins) + sign_idx * n_mag_bins + mag_bin
        # sign_idx: 0 for negative, 1 for positive
        self.tokens_per_prog = 2 * n_mag_bins
        
        self.special_offset = self.prog_offset + (n_programs * self.tokens_per_prog)
        
        self.bos_token = self.special_offset + 0
        self.eos_token = self.special_offset + 1
        self.endrec_token = self.special_offset + 2
        self.pad_token = self.special_offset + 3
        
        self.vocab_size = self.special_offset + 4

    def encode_record(
        self, 
        reg_idx: int, 
        dose_val: float, 
        deltaP: torch.Tensor, 
        top_p: int = 16,
        state_bin: int = 0
    ) -> torch.Tensor:
        """
        Encodes a single perturbation example into a token sequence.
        deltaP: [K] tensor of program activations
        """
        tokens = []
        
        # 1. Header tokens
        tokens.append(self.reg_offset + reg_idx)
        
        # Simple dose binning (assume dose is 0-1)
        dose_bin = min(int(dose_val * self.n_dose_bins), self.n_dose_bins - 1)
        tokens.append(self.dose_offset + dose_bin)
        
        tokens.append(self.state_offset + state_bin)
        
        # 2. Content tokens (Top-P programs)
        abs_p = torch.abs(deltaP)
        vals, indices = torch.topk(abs_p, k=min(top_p, self.n_programs))
        
        for i in range(len(indices)):
            p_idx = indices[i].item()
            val = deltaP[p_idx].item()
            
            sign_idx = 1 if val >= 0 else 0
            # Assume magnitude is normalized or typically < 10.0, use log or linear bins
            # For now, simple linear binning if we assume data is somewhat normalized
            mag_bin = min(int(abs(val) * self.n_mag_bins), self.n_mag_bins - 1)
            
            prog_token = self.prog_offset + (p_idx * self.tokens_per_prog) + (sign_idx * self.n_mag_bins) + mag_bin
            tokens.append(prog_token)
            
        # 3. Tail
        tokens.append(self.endrec_token)
        
        return torch.tensor(tokens, dtype=torch.long)

    def save_vocab(self, path: str):
        meta = {
            "n_regulators": self.n_regulators,
            "n_programs": self.n_programs,
            "n_dose_bins": self.n_dose_bins,
            "n_state_bins": self.n_state_bins,
            "n_mag_bins": self.n_mag_bins,
            "vocab_size": self.vocab_size,
            "offsets": {
                "reg": self.reg_offset,
                "dose": self.dose_offset,
                "state": self.state_offset,
                "prog": self.prog_offset,
                "special": self.special_offset
            }
        }
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2)
