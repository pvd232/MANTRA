# scripts/train_energy.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import argparse
import yaml
import torch
import scanpy as sc
import numpy as np

from mantra.config import EnergyModelConfig, EnergyTrainConfig
from mantra.eggfm.trainer import train_energy_model
from utils import subset_anndata

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train EGGFM energy model on K562")
    p.add_argument("--params", type=str, required=True,
                   help="YAML params file (must contain eggfm_model and eggfm_train)")
    p.add_argument("--ad", type=str, required=True,
                   help="Preprocessed K562 AnnData (e.g. data/interim/k562_replogle_prep.h5ad)")
    p.add_argument("--out", type=str, required=True,
                   help="Output directory for checkpoints (e.g. out/models/eggfm)")
    return p




def main() -> None:
    args = build_argparser().parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())

    model_cfg = EnergyModelConfig(**params.get("eggfm_model", {}))
    train_cfg = EnergyTrainConfig(**params.get("eggfm_train", {}))

    # 1) Load prepped K562 AnnData
    ad = sc.read_h5ad(args.ad)

    # 2) optional subsample for this experiment
    train_n_cells = params["eggfm_train"].get("n_cells_sample", None)
    if train_n_cells is not None:
        ad_prep = subset_anndata(ad, train_n_cells, random_state=params.get("seed", 0))
    else:
        ad_prep = ad
        
    # 3) restrict to HVGs if present
    if "highly_variable" in ad_prep.var:
        ad_prep = ad_prep[:, ad_prep.var["highly_variable"]].copy()
        print(f"Using HVGs only: n_vars = {ad_prep.n_vars}")

        # ---- NEW: further clamp HVGs to top N by dispersions_norm ----
        max_hvg = params["eggfm_train"].get("max_hvg", None)
        if max_hvg is not None and ad_prep.n_vars > max_hvg:
            if "dispersions_norm" in ad_prep.var:
                disp = ad_prep.var["dispersions_norm"].to_numpy()
                order = np.argsort(disp)[::-1]  # descending
            else:
                # fallback: arbitrary but deterministic
                order = np.arange(ad_prep.n_vars)

            keep_idx = order[:max_hvg]
            ad_prep = ad_prep[:, keep_idx].copy()
            print(
                f"Subsetting HVGs from {len(order)} → {ad_prep.n_vars} "
                f"(top {max_hvg} by dispersions_norm)"
            )
    else:
        print("No 'highly_variable' flag in ad.var; using all genes as-is.")
    
    # 4) Train energy model
    bundle = train_energy_model(
        ad_prep=ad_prep,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )

    energy_model = bundle.model
    mean = bundle.mean
    std = bundle.std
    var_names = bundle.feature_names

    # 5) Save checkpoint
    ckpt = {
        "state_dict": energy_model.state_dict(),
        "model_cfg": {
            "hidden_dims": list(model_cfg.hidden_dims),
        },
        "n_genes": energy_model.n_genes,
        "var_names": var_names,
        "mean": mean,
        "std": std,
        "space": bundle.space,
    }

    # Derive a short tag for the representation space, e.g. "expr", "pca", "phate"
    space_tag = str(bundle.space).replace("X_", "")  # if you ever use "X_pca", etc.
    n_genes = int(energy_model.n_genes)

    ckpt_name = f"eggfm_energy_k562_{space_tag}_hvg{n_genes}.pt"
    ckpt_path = out_dir / ckpt_name

    torch.save(ckpt, ckpt_path)
    print(f"Saved EGGFM energy checkpoint to {ckpt_path}", flush=True)



if __name__ == "__main__":
    main()
