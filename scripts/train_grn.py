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
from torch.utils.data import DataLoader

from mantra.config import (
    GRNModelConfig,
    GRNTrainConfig,
    GRNLossConfig,
)
from mantra.grn.dataset import K562RegDeltaDataset
from mantra.grn.models import GRNGNN, TraitHead
from mantra.grn.priors import build_energy_prior_from_ckpt
from mantra.grn.trainer import GRNTrainer


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train GRN GNN on K562 with pre-trained EGGFM energy prior"
    )

    p.add_argument(
        "--params",
        type=str,
        required=True,
        help="YAML params file (must contain grn_model, grn_train, grn_loss)",
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

    # ---- load AnnData ----
    qc_ad = sc.read_h5ad(args.ad)

    # ---- datasets ----
    train_ds = K562RegDeltaDataset(Path(args.train_npz))
    val_ds: Optional[K562RegDeltaDataset] = (
        K562RegDeltaDataset(Path(args.val_npz))
        if args.val_npz is not None
        else None
    )

    G = train_ds.n_genes
    n_regulators = train_ds.n_regulators

    # ---- adjacency ----
    if args.adj is not None:
        A_np = np.load(args.adj).astype(np.float32)
    else:
        A_np = np.eye(G, dtype=np.float32)
    A = torch.from_numpy(A_np).to(device)

    # ---- cNMF W ----
    if args.cnmf_W is not None:
        W_np = np.load(args.cnmf_W).astype(np.float32)  # [G,K]
    else:
        # identity: effectively disables program loss when lambda_prog > 0
        W_np = np.eye(G, dtype=np.float32)
    W = torch.from_numpy(W_np).to(device)

    # ---- reference state x_ref ----
    # We need x_ref in the SAME gene space as ΔE and the energy prior (G genes).
    # qc_ad currently has 2000 HVGs, so subset it to the 75 genes used by EGGFM/NPZ.

    # 1) Load HVG names from the energy checkpoint
    ckpt = torch.load(args.energy_ckpt, map_location="cpu")
    hvg_names = np.array(ckpt["var_names"])
    if hvg_names.shape[0] != G:
        raise ValueError(
            f"Energy ckpt var_names has {hvg_names.shape[0]} genes, "
            f"but ΔE has {G}. These must match."
        )

    # 2) Align qc_ad.var_names to this list
    var_names = np.array(qc_ad.var_names.astype(str))
    gene_to_idx = {g: i for i, g in enumerate(var_names)}

    missing = [g for g in hvg_names if g not in gene_to_idx]
    if missing:
        raise ValueError(
            f"Could not align qc_ad genes to energy ckpt/NPZ space: "
            f"{len(missing)} genes missing. Examples: {missing[:10]}"
        )

    idx = np.array([gene_to_idx[g] for g in hvg_names], dtype=int)
    qc_ad_sub = qc_ad[:, idx].copy()
    print(
        f"[align] subset qc_ad from {qc_ad.n_vars} → {qc_ad_sub.n_vars} genes "
        f"to match ΔE / energy prior space.",
        flush=True,
    )

    # 3) Compute x_ref in this aligned space
    X = qc_ad_sub.X
    if sp_sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    x_ref_np = X.mean(axis=0)  # [G]
    if x_ref_np.shape[0] != G:
        raise ValueError(
            f"Gene dimension mismatch after alignment: x_ref has {x_ref_np.shape[0]} genes, "
            f"but ΔE has {G}."
        )    
    x_ref = torch.from_numpy(x_ref_np).to(device)

    # ---- dataloaders ----
    train_loader = DataLoader(
        train_ds,
        batch_size=grn_train_cfg.batch_size,
        shuffle=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=grn_train_cfg.batch_size,
            shuffle=False,
        )

    # ---- energy prior (pretrained EGGFM) ----
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
        energy_prior=energy_prior,
        loss_cfg=grn_loss_cfg,
        train_cfg=grn_train_cfg,
        device=str(device),
    )

    trainer.fit(train_loader, val_loader)

    # ---- save best checkpoint ----
    ckpt = {
        "model_state_dict": trainer.best_model_state
            if trainer.best_model_state is not None
            else model.state_dict(),  # fallback: last state
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
        # --- prior metadata for later reload ---
        "prior_type": "eggfm_energy",
        "energy_ckpt_path": str(Path(args.energy_ckpt).resolve()),
        "energy_var_names": hvg_names,  # np.array/list of gene IDs in this space
    }

    ckpt_path = out_dir / "grn_k562_energy_prior.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Saved GRN checkpoint to {ckpt_path}", flush=True)


    ckpt_path = out_dir / "grn_k562_energy_prior.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Saved GRN checkpoint to {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
