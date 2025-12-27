import torch
from torch import nn, Tensor
from typing import Optional, Dict
import sys
from pathlib import Path

# Local imports
sys.path.append("/home/machina/MANTRA/src")
sys.path.append(str(Path(__file__).parent.parent.parent / "v4_recursive_injection/models"))
from v4_models import GeneGNNLayer_V4 as GeneGNNLayer_V5

class GRNGNN_V5(nn.Module):
    """
    Final MANTRA GNN with Recursive Multi-Scale Nexus Injection.
    """
    def __init__(
        self,
        n_regulators: int,
        n_genes: int,
        n_layers: int = 3,
        gene_emb_dim: int = 64,
        hidden_dim: int = 128,
        d_nexus: int = 256,
        dropout: float = 0.1,
        use_dose: bool = False,
    ) -> None:
        super().__init__()
        from mantra.grn.models import ConditionEncoder
        
        self.n_genes = n_genes
        self.use_dose = use_dose
        self.cond_encoder = ConditionEncoder(
            n_regulators=n_regulators,
            hidden_dim=hidden_dim,
            use_dose=use_dose
        )
        self.gene_emb = nn.Parameter(0.01 * torch.randn(n_genes, gene_emb_dim))
        
        self.layers = nn.ModuleList([
            GeneGNNLayer_V5(
                d_in=gene_emb_dim if i == 0 else hidden_dim,
                d_out=hidden_dim,
                d_cond=hidden_dim,
                d_nexus=d_nexus
            ) for i in range(n_layers)
        ])
        
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, reg_idx, dose, A, nexus_signal=None):
        cond = self.cond_encoder(reg_idx, dose)
        B = reg_idx.shape[0]
        h = self.gene_emb.unsqueeze(0).expand(B, self.n_genes, -1)
        
        # Recursive Injection: Pass nexus_signal into EACH layer
        for layer in self.layers:
            h = layer(h, cond, A, nexus_signal=nexus_signal)
            
        return self.readout(h).squeeze(-1)
