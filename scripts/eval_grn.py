#!/usr/bin/env python
# scripts/eval_grn.py

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from mantra.grn.dataset import K562RegDeltaDataset
from mantra.grn.models import GRNGNN, TraitHead
from mantra.grn.trainer import compute_grn_losses
from mantra.grn.priors import build_energy_prior_from_ckpt


def load_grn_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
):
    """
    Reconstruct GRN model, optional TraitHead, adjacency A, W, x_ref,
    energy_prior, and loss_cfg from a saved GRN checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # ----- basic config -----
    model_cfg = ckpt["grn_model_cfg"]          # dict
    loss_cfg_dict = ckpt["grn_loss_cfg"]       # dict
    n_regulators = int(ckpt["n_regulators"])
    n_genes = int(ckpt["n_genes"])

    # ----- core GRN model -----
    model = GRNGNN(
        n_regulators=n_regulators,
        n_genes=n_genes,
        n_layers=model_cfg.get("n_layers", 3),
        gene_emb_dim=model_cfg.get("gene_emb_dim", 64),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        dropout=model_cfg.get("dropout", 0.1),
        use_dose=model_cfg.get("use_dose", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ----- optional trait head -----
    trait_head = None
    n_traits = model_cfg.get("n_traits", 0)
    if n_traits > 0 and ckpt.get("trait_head_state_dict") is not None:
        W_np = ckpt["W"]
        K = W_np.shape[1]
        trait_head = TraitHead(
            n_programs=K,
            n_traits=n_traits,
            hidden_dim=model_cfg.get("trait_hidden_dim", 64),
        ).to(device)
        trait_head.load_state_dict(ckpt["trait_head_state_dict"])

    # ----- A, W, x_ref -----
    A_np = ckpt["A"].astype(np.float32)
    W_np = ckpt["W"].astype(np.float32)
    x_ref_np = ckpt["x_ref"].astype(np.float32)

    A = torch.from_numpy(A_np).to(device)
    W = torch.from_numpy(W_np).to(device)
    x_ref = torch.from_numpy(x_ref_np).to(device)

    # ----- energy prior -----
    energy_ckpt_path = ckpt["energy_ckpt_path"]
    energy_var_names = ckpt["energy_var_names"]
    energy_prior = build_energy_prior_from_ckpt(
        ckpt_path=energy_ckpt_path,
        gene_names=energy_var_names,
        device=device,
    )

    # ----- loss config -----
    # SimpleNamespace is enough; compute_grn_losses just reads attributes
    loss_cfg = SimpleNamespace(**loss_cfg_dict)

    return model, trait_head, A, W, x_ref, energy_prior, loss_cfg


def eval_split(
    name: str,
    npz_path: Path,
    model: GRNGNN,
    trait_head: TraitHead | None,
    A: torch.Tensor,
    x_ref: torch.Tensor,
    W: torch.Tensor,
    energy_prior: torch.nn.Module,
    loss_cfg,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    Evaluate a trained GRN on a given NPZ split and return
    averaged metrics: loss, L_expr, L_geo, L_prog, L_trait.
    """
    ds = K562RegDeltaDataset(npz_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    if trait_head is not None:
        trait_head.eval()

    totals = {
        "loss": 0.0,
        "L_expr": 0.0,
        "L_geo": 0.0,
        "L_prog": 0.0,
        "L_trait": 0.0,
    }
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = compute_grn_losses(
                model=model,
                A=A,
                batch=batch,
                x_ref=x_ref,
                energy_prior=energy_prior,
                W=W,
                loss_cfg=loss_cfg,
                trait_head=trait_head,
            )
            bsz = batch["reg_idx"].shape[0]
            n_samples += bsz

            totals["loss"] += out["loss"].item() * bsz
            totals["L_expr"] += out["L_expr"].item() * bsz
            totals["L_geo"] += out["L_geo"].item() * bsz
            totals["L_prog"] += out["L_prog"].item() * bsz
            totals["L_trait"] += out["L_trait"].item() * bsz

    metrics = {k: v / n_samples for k, v in totals.items()}
    print(f"[{name}] N={n_samples} | " +
          " ".join(f"{k}={metrics[k]:.4f}" for k in metrics))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate a trained GRN checkpoint on train/val NPZ splits."
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to GRN checkpoint (.pt), e.g. out/models/grn/hvg75/grn_k562_energy_prior.pt",
    )
    ap.add_argument(
        "--train-npz",
        type=str,
        required=True,
        help="Train NPZ path, e.g. data/interim/grn_k562_gwps_hvg75_npz/train.npz",
    )
    ap.add_argument(
        "--val-npz",
        type=str,
        default=None,
        help="Optional val NPZ path, e.g. data/interim/grn_k562_gwps_hvg75_npz/val.npz",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation",
    )

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, trait_head, A, W, x_ref, energy_prior, loss_cfg = load_grn_from_checkpoint(
        Path(args.ckpt),
        device=device,
    )

    # Train split metrics
    eval_split(
        "train",
        Path(args.train_npz),
        model,
        trait_head,
        A,
        x_ref,
        W,
        energy_prior,
        loss_cfg,
        device=device,
        batch_size=args.batch_size,
    )

    # Val split metrics (if provided)
    if args.val_npz is not None:
        eval_split(
            "val",
            Path(args.val_npz),
            model,
            trait_head,
            A,
            x_ref,
            W,
            energy_prior,
            loss_cfg,
            device=device,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
