# scripts/eggfm_train_k562.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import argparse
import yaml
import torch
import scanpy as sc

from mantra.eggfm.config import EnergyModelConfig, EnergyTrainConfig
from mantra.eggfm.train_energy import train_energy_model

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

    # 2) Train energy model
    bundle = train_energy_model(
        ad_prep=ad,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )

    energy_model = bundle.model
    mean = bundle.mean
    std = bundle.std
    var_names = bundle.feature_names

    # 3) Save checkpoint
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

    ckpt_path = out_dir / "eggfm_energy_k562_v1.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Saved EGGFM energy checkpoint to {ckpt_path}", flush=True)

if __name__ == "__main__":
    main()
