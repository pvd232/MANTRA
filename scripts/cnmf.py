#!/usr/bin/env python3
# src/mantra/scripts/cnmf.py
"""
Run consensus NMF (cNMF) on a QC'd AnnData object to obtain
gene-program loadings W [G, K] and related artifacts.

This script:
  - loads a QC'd AnnData
  - optionally aligns genes to an EGGFM energy checkpoint (for ΔE / NPZ compatibility)
  - builds a CNMFConfig from YAML + CLI overrides
  - runs cNMF
  - saves W_consensus, all per-run programs, cluster labels, run coverage, and manifest

Typical usage:

  python scripts/cnmf.py \
      --ad data/interim/k562_gwps_unperturbed_qc.h5ad \
      --out out/programs/k562_cnmf_hvg75 \
      --k 75 \
      --use-hvg-only \
      --n-restarts 20 \
      --seed 7 \
      --name k562_cnmf_hvg75 \
      --energy-ckpt out/models/eggfm/eggfm_energy_k562_hvg_hvg75.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import scanpy as sc
import torch
import yaml

from mantra.programs.cnmf import run_cnmf, save_cnmf_result
from mantra.programs.config import CNMFConfig


# ---------------------------------------------------------------------
# CLI / config glue
# ---------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run consensus NMF on a QC'd AnnData object to obtain "
            "gene-program loadings W [G,K] and related artifacts."
        )
    )

    p.add_argument(
        "--ad",
        type=str,
        required=True,
        help="Input QC AnnData (e.g. data/interim/k562_gwps_unperturbed_qc.h5ad)",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for W arrays and manifest.",
    )
    p.add_argument(
        "--params",
        type=str,
        default=None,
        help="Optional YAML params file; if provided, uses the 'cnmf' section as defaults.",
    )

    # core CNMF config (CLI overrides YAML)
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of programs / components (rank K). Overrides cnmf.n_components.",
    )
    p.add_argument(
        "--use-hvg-only",
        action="store_true",
        help="Restrict to ad.var[hvg_key]==True if present. Overrides cnmf.use_hvg_only.",
    )
    p.add_argument(
        "--n-restarts",
        type=int,
        default=None,
        help="Number of independent NMF runs. Overrides cnmf.n_restarts.",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Max iterations for NMF per run. Overrides cnmf.max_iter.",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Convergence tolerance for NMF. Overrides cnmf.tol.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Overall regularization strength for NMF. Overrides cnmf.alpha.",
    )
    p.add_argument(
        "--l1-ratio",
        type=float,
        default=None,
        help="Mix between L1 (1.0) and L2 (0.0). Overrides cnmf.l1_ratio.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for NMF + consensus. Overrides cnmf.random_state.",
    )
    p.add_argument(
        "--name",
        type=str,
        default="k562_cnmf",
        help="Short name used as prefix for output files.",
    )

    # optional alignment to EGGFM / NPZ gene space
    p.add_argument(
        "--energy-ckpt",
        type=str,
        default=None,
        help=(
            "Optional EGGFM checkpoint (.pt). If provided, subset AnnData to "
            "ckpt['var_names'] (or 'feature_names') so W rows align with ΔE / energy prior."
        ),
    )

    return p


def make_cfg(args: argparse.Namespace) -> CNMFConfig:
    """
    Merge YAML cnmf section (if present) with CLI overrides
    into a CNMFConfig dataclass.
    """
    base: Dict[str, Any] = {}
    if args.params is not None:
        base = yaml.safe_load(Path(args.params).read_text()).get("cnmf", {}) or {}

    default_cfg = CNMFConfig()

    def pick(field: str, cli_value, default):
        # CLI (if not None) overrides YAML; else YAML; else dataclass default
        if cli_value is not None:
            return cli_value
        if field in base:
            return base[field]
        return default

    cfg = CNMFConfig(
        n_components=pick("n_components", args.k, default_cfg.n_components),
        n_restarts=pick("n_restarts", args.n_restarts, default_cfg.n_restarts),
        max_iter=pick("max_iter", args.max_iter, default_cfg.max_iter),
        tol=pick("tol", args.tol, default_cfg.tol),
        use_hvg_only=pick("use_hvg_only", args.use_hvg_only, default_cfg.use_hvg_only),
        hvg_key=base.get("hvg_key", default_cfg.hvg_key),
        n_top_genes=base.get("n_top_genes", default_cfg.n_top_genes),
        min_cells_per_gene=base.get(
            "min_cells_per_gene", default_cfg.min_cells_per_gene
        ),
        scale_cells=base.get("scale_cells", default_cfg.scale_cells),
        alpha=pick("alpha", args.alpha, default_cfg.alpha),
        l1_ratio=pick("l1_ratio", args.l1_ratio, default_cfg.l1_ratio),
        consensus_kmeans_n_init=base.get(
            "consensus_kmeans_n_init", default_cfg.consensus_kmeans_n_init
        ),
        filter_by_coverage=base.get(
            "filter_by_coverage", default_cfg.filter_by_coverage
        ),
        min_run_coverage=base.get(
            "min_run_coverage", default_cfg.min_run_coverage
        ),
        random_state=pick("random_state", args.seed, default_cfg.random_state),
    )
    return cfg


# ---------------------------------------------------------------------
# Optional alignment to energy checkpoint gene space
# ---------------------------------------------------------------------


def subset_to_energy_genes(
    ad: sc.AnnData,
    energy_ckpt_path: str,
) -> Tuple[sc.AnnData, np.ndarray]:
    """
    Subset AnnData to the genes (and order) used by the EGGFM energy checkpoint.

    This ensures:
      - W rows == #genes in ΔE == len(ckpt.var_names)
      - ordering matches the ΔE / energy prior space.
    """
    print(f"[CNMF] Aligning genes to energy checkpoint: {energy_ckpt_path}", flush=True)
    ckpt = torch.load(energy_ckpt_path, map_location="cpu")
    if "var_names" in ckpt:
        gene_names = np.array(ckpt["var_names"])
    elif "feature_names" in ckpt:
        gene_names = np.array(ckpt["feature_names"])
    else:
        raise KeyError(
            "Energy checkpoint missing 'var_names'/'feature_names'; "
            "cannot align genes."
        )

    var_names = np.array(ad.var_names.astype(str))
    gene_to_idx = {g: i for i, g in enumerate(var_names)}

    missing = [g for g in gene_names if g not in gene_to_idx]
    if missing:
        raise ValueError(
            f"[CNMF] Could not align AnnData genes to energy ckpt space: "
            f"{len(missing)} missing. Examples: {missing[:10]}"
        )

    idx = np.array([gene_to_idx[g] for g in gene_names], dtype=int)
    ad_sub = ad[:, idx].copy()
    print(
        f"[CNMF] Subset AnnData from {ad.n_vars} → {ad_sub.n_vars} genes "
        f"to match energy ckpt ordering.",
        flush=True,
    )
    return ad_sub, gene_names


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = build_argparser().parse_args()

    ad_path = Path(args.ad)
    out_dir = Path(args.out)

    print(f"[CNMF] Loading AnnData from {ad_path} ...", flush=True)
    ad = sc.read_h5ad(ad_path)
    print(f"[CNMF] ad.n_obs={ad.n_obs}, ad.n_vars={ad.n_vars}", flush=True)

    # Optional alignment to EGGFM / NPZ gene space before running CNMF
    energy_genes: np.ndarray | None = None
    if args.energy_ckpt is not None:
        ad, energy_genes = subset_to_energy_genes(ad, args.energy_ckpt)
        print(
            f"[CNMF] Energy-aligned gene space: {energy_genes.shape[0]} genes",
            flush=True,
        )

    cfg = make_cfg(args)
    print(f"[CNMF] Config: {cfg}", flush=True)

    result = run_cnmf(ad, cfg)

    save_cnmf_result(out_dir, ad, result, prefix=args.name)

    if energy_genes is not None:
        genes_path = out_dir / f"{args.name}_genes_aligned.npy"
        np.save(genes_path, energy_genes)
        print(f"[CNMF] Saved aligned gene list to {genes_path}", flush=True)


if __name__ == "__main__":
    main()
