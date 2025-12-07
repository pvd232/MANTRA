# scripts/grn_train.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import argparse
import yaml
import numpy as np
import torch
import scanpy as sc
from scipy import sparse as sp_sparse

from mantra.config import (
    GRNModelConfig,
    GRNTrainConfig,
    GRNLossConfig,
    EnergyModelConfig,
    EnergyTrainConfig,
)
from mantra.grn.dataset import K562RegDeltaDataset
from mantra.grn.models import GRNGNN, TraitHead
from mantra.grn.priors import build_energy_prior_from_ckpt
from mantra.grn.trainer import GRNTrainer

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train GRN GNN on K562 with energy prior")

    p.add_argument(
        "--params",
        type=str,
        required=True,
        help="YAML params file (contains grn_model, grn_train, grn_loss, eggfm_model, eggfm_train)",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for GRN checkpoints / logs",
    )
    p.add_argument(
        "--ad",
        type=str,
        required=True,
        help="Preprocessed K562 AnnData (same gene order as ΔE)",
    )
    p.add_argument(
        "--train-npz",
        type=str,
        required=True,
        help="NPZ with aggregated (reg_idx, deltaE, deltaP_obs, deltaY_obs, dose)",
    )
    p.add_argument(
        "--val-npz",
        type=str,
        default=None,
        help="Optional NPZ for validation set",
    )
    p.add_argument(
        "--adj",
        type=str,
        default=None,
        help="Optional .npy adjacency [G,G]. If not provided, uses identity.",
    )
    p.add_argument(
        "--cnmf-W",
        type=str,
        default=None,
        help="Optional .npy cNMF loadings W [G,K]. If missing, uses identity [G,G].",
    )    
    p.add_argument(
        "--energy-ckpt",
        type=str,
        required=True,
        help="Path to pre-trained EGGFM energy checkpoint (.pt)",
    )
    return p

def main() -> None:
    args = build_argparser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load params ----
    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())

    grn_model_cfg = GRNModelConfig(**params.get("grn_model", {}))
    grn_train_cfg = GRNTrainConfig(**params.get("grn_train", {}))
    grn_loss_cfg = GRNLossConfig(**params.get("grn_loss", {}))

    energy_model_cfg = EnergyModelConfig(**params.get("eggfm_model", {}))
    energy_train_cfg = EnergyTrainConfig(**params.get("eggfm_train", {}))

    # ---- load data ----
    qc_ad = sc.read_h5ad(args.ad)

    train_ds = K562RegDeltaDataset(Path(args.train_npz))
    val_ds = (
        K562RegDeltaDataset(Path(args.val_npz))
        if args.val_npz is not None
        else None
    )

    G = train_ds.n_genes
    n_regulators = train_ds.n_regulators

    # ---- adjacency ----
    if args.adj is not None:
        A_np = np.load(args.adj)
    else:
        A_np = np.eye(G, dtype=np.float32)
    A = torch.from_numpy(A_np.astype(np.float32)).to(device)

    # ---- cNMF W ----
    if args.cnmf_W is not None:
        W_np = np.load(args.cnmf_W).astype(np.float32)  # [G,K]
    else:
        W_np = np.eye(G, dtype=np.float32)              # identity, effectively disables L_prog
    W = torch.from_numpy(W_np).to(device)

    # ---- reference state x_ref ----
    X = qc_ad.X
    if sp_sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    x_ref_np = X.mean(axis=0)  # [G], simple default; you can replace with controls-only mean
    if x_ref_np.shape[0] != G:
        raise ValueError(
            f"Gene dimension mismatch: x_ref has {x_ref_np.shape[0]} genes, "
            f"but deltaE has {G}."
        )

    x_ref = torch.from_numpy(x_ref_np).to(device)

    # ---- after you create train_ds / val_ds ----
    train_loader = DataLoader(train_ds, batch_size=grn_train_cfg.batch_size, shuffle=True)

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=grn_train_cfg.batch_size, shuffle=False)

    # ---- energy prior ----
    energy_prior = build_energy_prior_from_ckpt(
        ckpt_path=args.energy_ckpt,
        gene_names=qc_ad.var_names,
        device=device,
    )

    # ---- GRN model ----
    model = GRNGNN(
        n_regulators=n_regulators,
        n_genes=G,
        n_layers=grn_model_cfg.n_layers,
        gene_emb_dim=grn_model_cfg.gene_emb_dim,
        hidden_dim=grn_model_cfg.hidden_dim,
        dropout=grn_model_cfg.dropout,
        use_dose=grn_model_cfg.use_dose,
    ).to(device)

    # ---- optional trait head ----
    trait_head: Optional[TraitHead] = None
    if grn_model_cfg.n_traits > 0:
        K = W_np.shape[1]
        trait_head = TraitHead(
            n_programs=K,
            n_traits=grn_model_cfg.n_traits,
            hidden_dim=grn_model_cfg.trait_hidden_dim,
        ).to(device)

    # ---- trainer ----
    trainer = GRNTrainer(
        grn_model=model,
        trait_head=trait_head,
        A=A,
        x_ref=x_ref,
        W=W,
        energy_fn=energy_prior,
        loss_cfg=grn_loss_cfg,
        train_cfg=grn_train_cfg,
        device=str(device),
    )

    trainer.fit(train_loader, val_loader)
    # ---- save best checkpoint ----
    ckpt = {
        "model_state_dict": trainer.best_model_state,
        "trait_head_state_dict": (
            trainer.best_trait_state if trait_head is not None else None
        ),
        "grn_model_cfg": grn_model_cfg.__dict__,
        "grn_train_cfg": grn_train_cfg.__dict__,
        "grn_loss_cfg": grn_loss_cfg.__dict__,
        "n_regulators": n_regulators,
        "n_genes": G,
        "W": W_np,
        "A": A_np,
        "x_ref": x_ref_np,
        "prior_type": args.prior_type,
    }
    ckpt_path = out_dir / "grn_k562_energy_prior.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Saved GRN checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
